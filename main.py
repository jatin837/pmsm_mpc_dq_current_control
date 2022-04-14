class Controller:
    @classmethod
    def make(cls, controller_type, environment, **controller_kwargs):
        assert controller_type in _controllers.keys(), f'Controller {controller_type} unknown'
        controller = _controllers[controller_type](environment, **controller_kwargs)
        return controller

    def control(self, state, reference):
        pass

    def reset(self):
        pass


class MPC(Controller):
    def __init__(self, environment, ph=5, ref_idx_q=0, ref_idx_d=1):
        # conversion of the coordinate systems
        t32 = environment.physical_system.electrical_motor.t_32
        q = environment.physical_system.electrical_motor.q
        self._backward_transformation = (lambda quantities, eps: t32(q(quantities[::-1], eps)))
    
        # indices
        self.ref_idx_i_q = ref_idx_q
        self.ref_idx_i_d = ref_idx_d
        self.i_sd_idx = environment.state_names.index('i_sd')
        self.i_sq_idx = environment.state_names.index('i_sq')
        self.u_a_idx = environment.state_names.index('u_a')
        self.u_b_idx = environment.state_names.index('u_b')
        self.u_c_idx = environment.state_names.index('u_c')
        self.u_sq_idx = environment.state_names.index('u_sq')
        self.u_sd_idx = environment.state_names.index('u_sd')
        self.omega_idx = environment.state_names.index('omega')
        self.epsilon_idx = environment.state_names.index('epsilon')

        # motor parameters
        self.tau = environment.physical_system.tau
        self.limits = environment.physical_system.limits
        self.l_q = environment.physical_system.electrical_motor.motor_parameter['l_q']
        self.l_d = environment.physical_system.electrical_motor.motor_parameter['l_d']
        self.psi_ = environment.physical_system.electrical_motor.motor_parameter['psi_p']
        self.r_s = environment.physical_system.electrical_motor.motor_parameter['r_s']
        self.p = environment.physical_system.electrical_motor.motor_parameter['p']
        self.ph_ = ph

    def control(self, state, reference):
        # initialize variables
        epsilon_el = state[self.epsilon_idx] * self.limits[self.epsilon_idx]
        omega = self.p * state[self.omega_idx] * self.limits[self.omega_idx]

        ref_q = []
        ref_d = []
        eps = []
        lim_a_up = []
        lim_a_low = []
        
        for i in range(self.ph_):
            ref_q.append(reference[self.ref_idx_i_q] * self.limits[self.i_sq_idx])
            ref_d.append(reference[self.ref_idx_i_d] * self.limits[self.i_sd_idx])
         
            eps.append(epsilon_el + (i-1) * self.tau * omega)
            lim_a_up.append(2 * self.limits[self.u_a_idx])
            lim_a_low.append(-2 * self.limits[self.u_a_idx])
        
        m = GEKKO(remote=False)
        
        # defenition of the prediction Horizon
        m.time = np.linspace(self.tau, self.tau * self.ph_, self.ph_)

        # defenition of the variables
        u_d = m.MV(value=state[self.u_sd_idx] * self.limits[self.u_sd_idx])
        u_q = m.MV(value=state[self.u_sq_idx] * self.limits[self.u_sq_idx])
        u_d.STATUS = 1
        u_q.STATUS = 1

        u_a_lim_up = m.Param(value=lim_a_up)
        u_a_lim_low = m.Param(value=lim_a_low)
        sq3 = math.sqrt(3)

        i_d = m.SV(value=state[self.i_sd_idx] * self.limits[self.i_sd_idx], lb=-self.limits[self.i_sd_idx], ub=self.limits[self.i_sd_idx] )
        i_q = m.SV(value=state[self.i_sq_idx] * self.limits[self.i_sq_idx], lb=-self.limits[self.i_sq_idx], ub=self.limits[self.i_sq_idx])

        epsilon = m.Param(value=eps)
        
        # reference trajectory
        traj_d = m.Param(value=ref_d)
        traj_q = m.Param(value=ref_q)
        
        # defenition of the constants
        omega = m.Const(value=omega)
        psi = m.Const(value=self.psi_)
        rs = m.Const(value=self.r_s)
        ld = m.Const(value=self.l_d)
        lq = m.Const(value=self.l_q)
        
        # control error
        e_d = m.CV()
        e_q = m.CV()
        e_d.STATUS = 1
        e_q.STATUS = 1
        
        # solver options
        m.options.CV_TYPE = 2
        m.options.IMODE = 6
        m.options.solver = 3
        m.options.WEB = 0
        m.options.NODES = 2
        
        # differential equations
        m.Equations([ld * i_d.dt() == u_d - rs * i_d + omega * lq * i_q,
                     lq * i_q.dt() == u_q - rs * i_q - omega * ld * i_d - omega * psi])
        
        # cost function
        m.Equations([e_d == (i_d - traj_d), e_q == (i_q - traj_q)])
        
        # voltage limitations
        m.Equation(u_a_lim_up >= 3/2 * m.cos(epsilon) * u_d - 3/2 * m.sin(epsilon) * u_q - sq3/2 * m.sin(epsilon) * u_d - sq3/2 * m.cos(epsilon) * u_q)
        m.Equation(u_a_lim_low <= 3 / 2 * m.cos(epsilon) * u_d - 3 / 2 * m.sin(epsilon) * u_q - sq3 / 2 * m.sin(epsilon) * u_d - sq3 / 2 * m.cos(epsilon) * u_q)
        m.Equation(u_a_lim_up >= sq3 * m.sin(epsilon) * u_d + sq3 * m.cos(epsilon) * u_q)
        m.Equation(u_a_lim_low <= sq3 * m.sin(epsilon) * u_d + sq3 * m.cos(epsilon) * u_q)
        m.Equation(u_a_lim_up >= -3 / 2 * m.cos(epsilon) * u_d + 3 / 2 * m.sin(epsilon) * u_q - sq3 / 2 * m.sin(epsilon) * u_d - sq3 / 2 * m.cos(epsilon) * u_q)
        m.Equation(u_a_lim_low <= -3 / 2 * m.cos(epsilon) * u_d + 3 / 2 * m.sin(epsilon) * u_q - sq3 / 2 * m.sin(epsilon) * u_d - sq3 / 2 * m.cos(epsilon) * u_q)
        
        # object to minimize
        m.Obj(e_d)
        m.Obj(e_q)
        
        # solving optimization problem
        m.solve(disp=False)
        
        # additional voltage limitation
        u_a, u_b, u_c = self._backward_transformation((u_q.NEWVAL, u_d.NEWVAL), epsilon_el)
        u_max = max(np.absolute(u_a - u_b), np.absolute(u_b - u_c), np.absolute(u_c - u_a))
        if u_max >= 2 * self.limits[self.u_a_idx]:
            u_a = u_a / u_max * 2 * self.limits[self.u_a_idx]
            u_b = u_b / u_max * 2 * self.limits[self.u_a_idx]
            u_c = u_c / u_max * 2 * self.limits[self.u_a_idx]
        
        # Zero Point Shift
        u_0 = 0.5 * (max(u_a, u_b, u_c) + min(u_a, u_b, u_c))
        u_a -= u_0
        u_b -= u_0
        u_c -= u_0
        
        # normalization of the manipulated variables
        u_a /= self.limits[self.u_a_idx]
        u_b /= self.limits[self.u_b_idx]
        u_c /= self.limits[self.u_c_idx]
        
        return u_a, u_b, u_c

    def reset(self):
        None

_controllers = {
    'mpc': MPC
}

