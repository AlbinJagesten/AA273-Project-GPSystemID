function new_y = dynamics(y,u,dt)

    new_y = y-dt*tanh(y^3+u);

end