# Reentry

The software within this repository aims to plot & simulate various atmospheric reentry from upper space, i.e. balistic reentry, lifting reentry, orbital decay.
It tries to be a first-handed simulation model so that one can get quickly the few parameters needed to characterize such a flight.
For now it is not available online, but I'm working on it.

Next versions features:
- 3D model & animation - might be a javascript or C+ step by step animation, with several center of view & zoom
- mass differential equations, to take account of the loss of mass as with the heat shield ablation.
- temperature & heat equations
- Earth / planets rotation
- several settings for several planets (g0, R, approximation of atmosphere's density)
- Monte Carlo analysis to find the zone (instead of point) for the crash (balistic reentry)
- Improve the whole rapidity & accuracy
