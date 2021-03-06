%addPath('/Volumes/Data/Projects/PoleBalance/mpgwrite/');

cart_width = .5;
cart_height = .3;
pole_length = .5;
wall_size = .1;

figure('color','white');

[generation, fitness, k, l, m, n, time_step, fail_position, fail_angle] = textread('./header.txt', '%d %f %d %d %d %d %f %f %f', 'headerlines', 1);

fps = 1 / time_step / 1000; % Hz

title({['Generation: ', num2str(generation)], ['Fitness: ', num2str(fitness)]});

% hide Y axis
set(gca, 'YTickLabelMode', 'Manual')
set(gca, 'YTick', [])
set(gca, 'YColor', [1 1 1])

axis ([(-fail_position - cart_width / 2 - 2 * wall_size) (fail_position + cart_width / 2 + 2 * wall_size) (-fail_position - cart_width / 2) (fail_position + cart_width / 2)]);

[cart_position, pole_angle, force] = textread('./data.txt', '%f %f %f', 'headerlines', 1);

rectangle('Position', [-fail_position - cart_width / 2, -wall_size, fail_position * 2 + cart_width, wall_size], 'faceColor', [.5, .5, .5], 'edgeColor', [.5, .5, .5]);
border_left = rectangle('Position', [-fail_position - cart_width / 2 - wall_size, -wall_size, wall_size, .6], 'faceColor', [.5, .5, .5], 'edgeColor', [.5, .5, .5]);
border_right = rectangle('Position', [+fail_position + cart_width / 2, -wall_size, wall_size, .6], 'faceColor', [.5, .5, .5], 'edgeColor', [.5, .5, .5]);

Frames = moviein(length(cart_position));

hold on;

cart = rectangle();
pole = line([0 0], [1 1], 'color', [.4 .7 .3], 'lineWidth', 2);
label_time = text(-2, -1.4, 'time');
label_angle = text(-2, -1.7, 'angle');
label_force = text(-2, -2, 'force');

for j = 1:length(cart_position)
    pole_angle(j) = pole_angle(j) + pi / 2;
    set(cart,'Position', [(cart_position(j) - cart_width / 2), 0, cart_width, cart_height]);
    set(pole,'XData', [cart_position(j), (cart_position(j) + cos(pole_angle(j)))]);
    set(pole,'YData', [(cart_height / 2), ((cart_height / 2) + sin(pole_angle(j)))]);
    
    if cart_position(j) == -fail_position
        set(border_left, 'faceColor', [.7, .1, .1]);
    elseif cart_position(j) == fail_position
        set(border_right, 'faceColor', [.7, .1, .1]);
    end
    
    set(label_time, 'String', strcat({'time:  '}, num2str(j * time_step, '%d'), ' msec'));
    set(label_angle, 'String', strcat({'angle:  '}, num2str((pole_angle(j) - pi / 2) * 180 / pi, '%.3f'), '^o'));
    set(label_force, 'String', strcat({'force:  '}, num2str(force(j), '%.3f'), {' N'}));
       
    refreshdata;
    drawnow;
    Frames(:, j) = getframe;
end

movie2avi(Frames, './build/Debug/render.avi', 'fps', fps, 'quality', 75);
