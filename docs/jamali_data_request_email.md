# Draft: data-request email to the Jamali Lab

**To:** corresponding author of the LEONARDO paper (Nat. Commun. 16, 2025, s41467-025-61632-1)
— find the corresponding-author address on the paper's title page / "Corresponding authors" section.
**Cc:** your thesis advisor (optional, adds credibility).
**Subject:** Request for raw LPTEM videos underlying LEONARDO — MSc thesis extending your simulator

---

Dear Prof. Jamali,

I am an MSc Data Science student at FINKI (Ss. Cyril and Methodius University, Skopje), working
on a thesis that builds a self-supervised latent video "world model" for liquid-phase TEM. Your
group's work is central to it: I pretrain on the SAM-EM synthetic dataset, extend the SAM-EM
simulator with particle growth/dissolution/coalescence dynamics, and benchmark predicted motion
statistics against LEONARDO.

To validate my model's forecasts against real dynamics — rather than only against simulated
ground truth — I would like to compare predicted trajectories and mean-squared-displacement
curves to the actual recordings. Would you be willing to share the **raw LPTEM video recordings**
underlying the LEONARDO trajectory dataset?

I would of course cite and acknowledge your group fully, use the data solely for this academic
thesis, and am happy to share my results and the extended simulator back with you.

Thank you very much for considering this, and for making SAM-EM and LEONARDO openly available —
they have been invaluable.

Best regards,
Nikola Jankovikj
MSc Data Science, FINKI — Ss. Cyril and Methodius University, Skopje
[your email] · [advisor name, if cc'd]

---

**Notes before sending:**
- Confirm the corresponding author + address from the paper itself (don't guess).
- This is **off the critical path** — the thesis stands on simulator ground truth + the public
  experimental clips (Au nanorods, aggregation) we already have. The raw videos are an *upgrade*
  for Step 13 (independent real-motion GT). Send it now purely because reply latency is months.
