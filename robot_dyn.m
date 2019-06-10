function x_next = robot_dyn(x,u,dt)

    x_next = x + dt * [sin(u);
                       cos(u)];

end