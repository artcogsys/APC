CURRENT MODEL:

- MODEL RECEIVES FOVEAL INPUT ONLY
- MODEL MINIMIZES DIFFERENCE BETWEEN INTERNAL REPRESENTATION AT t-1 and FOVEAL INPUT AT t
- MODEL USES PERIPHERAL REPRESENTATION TO DECIDE ON NEXT SACCADE (BLURRED VERSION OF HIGH RES REPRESENTATION)
- MODEL RUNS ON SINGLE GPU
To do:
- werk aan GPU implementatie (werkt op single GPU)
- Testen op video input (test_video startpunt; test color representations)
- Toevoegen van meerdere lagen (a la Cox) => leren we voorspellen beyond de geziene foveations?

- add physical constraints (saccade speed cf. frame speed)
- performance testing; dependence on various parameters
- compare with biological vision; saccade patterns; development of saccade patterns etc
- think harder about sensory constraints (organization of retina etc; and what we can learn)
- can we have topography emerge by assuming distance constraints (conflicts with CNN though)

