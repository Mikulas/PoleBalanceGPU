
/**
 * c_ READ ONLY constants for computing force
 * fitness WRITE READ
 */
__kernel void add(__global int *c_c_position, __global int *c_c_velocity, __global int *c_p_angle, __global int *c_p_velocity, __global int *fitness)
{
	const float PI = 3.14159265;
	const int gid = get_global_id(0);

	const float fail_position = 2.4; // [meters]
	const float fail_angle = PI / 6; // [radians] 30 degrees
	const float g_acceleration = -9.81; // [meters/second/second]
	const float abs_force = 1; // [newtons]
	const float p_length = 0.5; // [meters] relative from pivot
	const float p_mass = 0.1; // [kilograms]
	const float c_mass = 1; // [kilograms]
	const int time_total = 60000; // [miliseconds]
	const int time_step = 25; // [miliseconds]

	/** default values */
	float c_position = 0;
	float c_velocity = 0;
	float c_acceleration = 0;
	float p_angle = PI / 36; // 5 degrees
	float p_velocity = 0;
	float p_acceleration = 0;
	float force;

	float delta = ((float)time_step / (float)1000);

	int t;
	for (t = 0; t < time_total; t += time_step) {
		// http://www.profjrwhite.com/system_dynamics/sdyn/s7/s7invp2/s7invp2.html (7.61, 7.62)
		c_position = c_position + delta * c_velocity;
		p_angle = p_angle + delta * p_velocity;

		force = abs_force * ((c_c_position[gid] * c_position + c_c_velocity[gid] * c_velocity + c_p_angle[gid] * p_angle + c_p_velocity[gid] * p_velocity) > 0 ? 1 : -1); // intentionally not signum, cart must always move
		c_acceleration = (force + p_mass * p_length * sin(p_angle) * p_velocity * p_velocity - p_mass * g_acceleration * cos(p_angle) * sin(p_angle)) / (c_mass + p_mass - p_mass * cos(p_angle) * cos(p_angle));
		p_acceleration = (force * cos(p_angle) - g_acceleration * (c_mass + p_mass) * sin(p_angle) + p_mass * p_length * cos(p_angle) * sin(p_angle) * p_velocity) / (p_mass * p_length * cos(p_angle) * cos(p_angle) - (c_mass + p_mass) * p_length);

		c_velocity = c_velocity + delta * c_acceleration;
		p_velocity = p_velocity - delta * p_acceleration; // why minus?

		if (c_position * (c_position > 0 ? 1 : -1) >= fail_position || p_angle * (p_angle > 0 ? 1 : -1) > fail_angle) {
			break;
		}
	}
	fitness[gid] = t;
}
