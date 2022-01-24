# Double-Pendulum
Simulation of the double pendulum using RK4
This is a program that simulates the double pendulum based on different (random) initial positions using Python.

I've always wanted to code this simulation up since I first made a physical model in my Physics course. It's been about 4 years since
then and my interest in programming has been increasing. I thought this would be a good project to do after solving the single pendulum equations
using RK4. For this simulation, I have made the line plot and scatter plot updates for each frame A LOT more efficient by using Matplotlibs
set_data() and set_array() functions instead of replotting each entire frame. I've also used queues, which operate on a first-in first-out (FIFO)
principle. These happen to be perfect to use for the tracer to help visualize the path of the pendulum.

Numpy and Matplotlib were used for matrix algebra and visualization.
The equations of motion were derived by hand using the Lagrangian Method and solved numerically using the 4th order Runge-Kutta method.

To see the magic, simply download the code and make sure all applicable libraries are installed and hit run.

<img width="603" alt="Screen Shot 2022-01-24 at 9 15 37 AM" src="https://user-images.githubusercontent.com/22279983/150799175-61f152dc-4933-4137-aea4-aac2d89d0da0.png">
