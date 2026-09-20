"""PCA-specific PSD noise editing for SDXL-Turbo (and future FLUX codecs).

Everything under this package is additive: it reads/imports/calls the sibling
``noise_init/`` package but never modifies it. See docs/ for the research plan
and implementation checklist.
"""
