function new_y = dynamics(y,u,dt)

    new_y = y-dt*[tanh(y(1)+u^3)];

end