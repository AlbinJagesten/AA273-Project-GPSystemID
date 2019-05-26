function new_y = dynamics(y,u,dt)

    new_y = y-dt*[tanh(y(1)^3+u);0];

end