Mission log Sept 23rd

Here is the log for the mission today:

Mission started: 8:00
Mission Ended: 12:30

Mission Objective:
- Set up prior
- Test adaptive algorithm

Mission Details:
- Pre_surveyor was initiated twice and aborted at some point where it stopped running. The time it stopped is roughly the same as the time it was stopped for the other missions.
- The Pre_surveyor produced the corrected prior based on the data it collected. The corrected prior is then used for the adaptive mission.
- The adaptive mission was started, and the AUV stopped running after it was travelling on the way to the starting location. Therefore, no more adaptive steps were conducted.

Possible errors:
- In the simulation phase, lauv-xplore-1 DUNE simulator was used. Everything was tested and verified, and it demonstrated its .
- The possible reason might be due to the different version of DUNE. Sine in the simulation, DUNE is not controlling the actual AUV to move, while in the actual testing, DUNE is controlling all of those components.

Mission Conclusion:
- The adaptive framework is only tested once, which seems not very sufficient. But future testings are expected in the coming fall.

Possible problems:
- SMS issues, time_out from the cache
- Satellite signal issues,

Future improvements
- Take a CTD with the boat to measure the locational salinity for validation.
