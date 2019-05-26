function x_next = robot_dyn(x,u,dt)

    x_next = x + dt * [u(1)*cos(x(3));
                       u(1)*sin(x(3));
                       u(2)];

end