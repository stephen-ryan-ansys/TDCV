function save_trajectory(filepath, frames, orientations, positions)
format_str = strcat("%d ", repmat('% .4f ', 1, 7), '\n');
f = fopen(filepath, 'w');
for i=1:length(frames)
    timestamp = frames(i);
    tx = positions(1, 1, i);
    ty = positions(1, 2, i);
    tz = positions(1, 3, i);
    q = -rotation_matrix_to_quaternion(orientations(:, :, i));
    qx = q(1);
    qy = q(2);
    qz = q(3);
    qw = q(4);
    fprintf(f, format_str, timestamp, tx, ty, tz, qx, qy, qz, qw);
end
fclose(f);
end
