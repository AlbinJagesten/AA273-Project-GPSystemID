function x_next = robot_dyn(x,u,dt)

    x_next = x + dt * [cos(u);
                       sin(u)];

end