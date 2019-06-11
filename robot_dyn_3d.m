function x_next = robot_dyn(x,u,dt)

    x_next = x + dt * [sin(x(3));
                       cos(x(3))
                       u];

end