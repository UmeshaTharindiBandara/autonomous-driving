"""
PID Controller for steering control
"""

import time


class PIDController:
    """PID controller with rate limiting and output clamping"""
    
    def __init__(self, kp, ki, kd, i_limit=0.6, rate_limit=0.03, out_limit=0.25, sign=-1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_limit = i_limit
        self.rate_limit = rate_limit
        self.out_limit = out_limit
        self.sign = sign
        
        self.i_term = 0.0
        self.prev_e = None
        self.prev_t = None
    
    def step(self, error, now=None, last_out=None):
        if now is None:
            now = time.time()
        
        if self.prev_t is None:
            self.prev_t = now
            self.prev_e = error
            u = self.kp * error
        else:
            dt = max(1e-3, now - self.prev_t)
            p = self.kp * error
            self.i_term += self.ki * error * dt
            self.i_term = max(-self.i_limit, min(self.i_limit, self.i_term))
            d = self.kd * (error - self.prev_e) / dt
            u = p + self.i_term + d
            self.prev_e = error
            self.prev_t = now
        
        u = -u * self.sign
        u = max(-self.out_limit, min(self.out_limit, u))
        
        if last_out is not None:
            delta = u - last_out
            if delta > self.rate_limit:
                u = last_out + self.rate_limit
            elif delta < -self.rate_limit:
                u = last_out - self.rate_limit
        
        return u
    
    def reset(self):
        self.i_term = 0.0
        self.prev_e = None
        self.prev_t = None