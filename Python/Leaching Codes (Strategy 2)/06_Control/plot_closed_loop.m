% plot_closed_loop.m  -- PGML closed-loop Se(z,t) surface
% Mirrors the Figure 1 block of the MATLAB Sim_column.m.
load('closed_loop_surface.mat');
[T_full, Z_full] = meshgrid(t_days, Z_vect);
figure
surf(T_full, Z_full, Se_full_mat, 'EdgeColor', 'none');
xlabel('Time (days)')
ylabel('Axial position z (cm)')
zlabel('Effective saturation S_e (-)')
title('S_e(z,t) -- PGML closed loop')
grid on
colorbar
