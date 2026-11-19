from __future__ import annotations

import torch
import torch.nn as nn


class ImageryAlignmentAdapter(nn.Module):
    """Map imagery fMRI voxels into the NSD perception voxel space before BrainVAE encoding.

    Two construction paths, both supported simultaneously for backward
    compatibility:

      1. Single-subject (legacy):
            ImageryAlignmentAdapter(imagery_voxels=15724, nsd_voxels=15724, ...)
         Behaves exactly like the original adapter; no subject embedding, no
         per-subject input layers. Existing checkpoints load as-is.

      2. Multi-subject (new):
            ImageryAlignmentAdapter(
                nsd_voxels=15724,
                voxels_per_subject={1: 15724, 2: 14278, ...},
                num_subjects=8,
                subject_embed_dim=64,
            )
         Each subject gets its own Linear(n_voxels_subject -> hidden_dim)
         front-end; downstream layers are shared and conditioned on a learnable
         subject embedding.
    """

    def __init__(
        self,
        imagery_voxels: int | None = None,
        nsd_voxels: int | None = None,
        hidden_dim: int = 2048,
        dropout: float = 0.0,
        num_subjects: int = 0,
        subject_embed_dim: int = 64,
        voxels_per_subject: dict[int, int] | None = None,
    ) -> None:
        super().__init__()
        if nsd_voxels is None:
            raise ValueError("nsd_voxels is required")
        self.nsd_voxels = nsd_voxels
        self.hidden_dim = hidden_dim
        self.num_subjects = num_subjects
        self.subject_embed_dim = subject_embed_dim if num_subjects > 0 else 0

        if voxels_per_subject:
            self.voxels_per_subject: dict[int, int] = {int(k): int(v) for k, v in voxels_per_subject.items()}
            self.multi_subject = True
        else:
            if imagery_voxels is None:
                raise ValueError("Provide either imagery_voxels (legacy) or voxels_per_subject (multi-subject)")
            self.imagery_voxels = int(imagery_voxels)
            self.voxels_per_subject = {}
            self.multi_subject = False

        if self.multi_subject:
            self.subject_input = nn.ModuleDict({
                str(sid): nn.Linear(int(n_voxels), hidden_dim)
                for sid, n_voxels in self.voxels_per_subject.items()
            })
            self.shared = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, nsd_voxels),
            )
            if num_subjects > 0:
                self.subject_embedding = nn.Embedding(num_subjects, subject_embed_dim)
                nn.init.normal_(self.subject_embedding.weight, std=0.02)
                self.subject_to_hidden = nn.Linear(subject_embed_dim, hidden_dim)
            else:
                self.subject_embedding = None
                self.subject_to_hidden = None
            # legacy attributes left undefined
        else:
            # ===== legacy single-subject path: matches the original adapter exactly =====
            in_dim = self.imagery_voxels + (subject_embed_dim if num_subjects > 0 else 0)
            if num_subjects > 0:
                self.subject_embedding = nn.Embedding(num_subjects, subject_embed_dim)
                nn.init.normal_(self.subject_embedding.weight, std=0.02)
            else:
                self.subject_embedding = None
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, nsd_voxels),
            )

    def forward(
        self,
        imagery_fmri,
        subject_id: torch.Tensor | None = None,
        raw_subject_ids: torch.Tensor | list[int] | None = None,
    ) -> torch.Tensor:
        if not self.multi_subject:
            # Legacy path
            if imagery_fmri.ndim != 2:
                imagery_fmri = imagery_fmri.flatten(1)
            if self.subject_embedding is not None:
                if subject_id is None:
                    raise ValueError("subject_id is required when num_subjects > 0")
                subj = self.subject_embedding(subject_id.long().to(imagery_fmri.device))
                imagery_fmri = torch.cat([imagery_fmri, subj], dim=-1)
            return self.net(imagery_fmri)

        # Multi-subject path: project per-subject, then shared backbone
        if isinstance(imagery_fmri, list):
            if raw_subject_ids is None:
                raise ValueError("raw_subject_ids is required when imagery_fmri is a list")
            ids_list = raw_subject_ids if isinstance(raw_subject_ids, list) else raw_subject_ids.tolist()
            projected = []
            for fmri, sid in zip(imagery_fmri, ids_list):
                if fmri.ndim == 1:
                    fmri = fmri.unsqueeze(0)
                projected.append(self.subject_input[str(int(sid))](fmri))
            hidden = torch.cat(projected, dim=0)
        else:
            if imagery_fmri.ndim != 2:
                imagery_fmri = imagery_fmri.flatten(1)
            if len(self.subject_input) == 1:
                only_sid = next(iter(self.subject_input))
                hidden = self.subject_input[only_sid](imagery_fmri)
            else:
                if raw_subject_ids is None:
                    raise ValueError("raw_subject_ids is required for multi-subject batched tensor input")
                ids_list = raw_subject_ids if isinstance(raw_subject_ids, list) else raw_subject_ids.tolist()
                rows = []
                for row, sid in zip(imagery_fmri, ids_list):
                    rows.append(self.subject_input[str(int(sid))](row.unsqueeze(0)))
                hidden = torch.cat(rows, dim=0)

        if self.subject_embedding is not None:
            if subject_id is None:
                raise ValueError("subject_id is required when num_subjects > 0")
            subj = self.subject_embedding(subject_id.long().to(hidden.device))
            hidden = hidden + self.subject_to_hidden(subj)

        return self.shared(hidden)
