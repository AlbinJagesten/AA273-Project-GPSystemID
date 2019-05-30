function x_next = robot_dyn(x,u,dt)

    x_next = x + dt * [cos(x(3));
                       sin(x(3));
                       u];

end