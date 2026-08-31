The data for the target drone is taken from two datasets: NeuroBEM (Small / Highly Maneuverable Drones) and Mid-Air (Medium-Sized / Less Maneuverable Drones).


Plan for the future:
* Physics will be added to the current guidance outputs so that the interceptor is not weightless
* After this, the pursuit trajectories will be recorded.
* Then the net flight dynamics will be added to easily calculate whether the net would reach a target based on recorded points
* Then, I am not sure whether to follow an IL or an RL path. This will have to be decided in the future.


To cite:
```
@article{bauersfeld2021neurobem,
  title={NeuroBEM: Hybrid Aerodynamic Quadrotor Model},
  author={Bauersfeld, Leonard and Kaufmann, Elia and Foehn, Philipp and Sun, Sihao and Scaramuzza, Davide},
  journal={RSS: Robotics, Science, and Systems},
  year={2021},
  publisher={IEEE}
}

@INPROCEEDINGS{Fonder2019MidAir,
author = {Michael Fonder and Marc Van Droogenbroeck},
title = {Mid-Air: A multi-modal dataset for extremely low altitude drone flights},
booktitle = {Conference on Computer Vision and Pattern Recognition Workshop (CVPRW)},
year = {2019},
month = {June}
}
```