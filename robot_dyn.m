function x_next = robot_dyn(x,u,dt)

    x_next = x + dt * [u;
                       u];

end