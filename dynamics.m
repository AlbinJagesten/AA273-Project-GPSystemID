function new_y = dynamics(y,u,dt)

    new_y = y-dt*tanh(y+u^3);

end