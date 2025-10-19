import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons
import math

# Logic


def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2)
    k4 = f(t + dt, y + dt*k3)
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# SYS


class SimplePendulum:
    def __init__(self, L=1.0, m=1.0, g=9.81, damping=0.0):
        self.L, self.m, self.g, self.damping = L, m, g, damping

    def derivs(self, t, y):
        θ, ω = y
        return np.array([ω, -(self.g/self.L)*math.sin(θ) - self.damping*ω])

    def energy(self, y):
        θ, ω = y
        KE = 0.5*self.m*(self.L*ω)**2
        PE = self.m*self.g*self.L*(1 - math.cos(θ))
        return KE, PE

    def coords(self, y):
        θ = y[0]
        return self.L*math.sin(θ), -self.L*math.cos(θ)


class DoublePendulum:
    def __init__(self, m1=1, m2=1, L1=1, L2=1, g=9.81, damping=0):
        self.m1, self.m2, self.L1, self.L2, self.g, self.damping = m1, m2, L1, L2, g, damping

    def derivs(self, t, y):
        th1, w1, th2, w2 = y
        m1, m2, L1, L2, g = self.m1, self.m2, self.L1, self.L2, self.g
        δ = th2 - th1
        den1 = (m1+m2)*L1 - m2*L1*math.cos(δ)**2
        den2 = (L2/L1)*den1
        domega1 = (m2*L1*w1*w1*math.sin(δ)*math.cos(δ)
                   + m2*g*math.sin(th2)*math.cos(δ)
                   + m2*L2*w2*w2*math.sin(δ)
                   - (m1+m2)*g*math.sin(th1)) / den1
        domega2 = (-m2*L2*w2*w2*math.sin(δ)*math.cos(δ)
                   + (m1+m2)*g*math.sin(th1)*math.cos(δ)
                   - (m1+m2)*L1*w1*w1*math.sin(δ)
                   - (m1+m2)*g*math.sin(th2)) / den2
        return np.array([w1, domega1 - self.damping*w1, w2, domega2 - self.damping*w2])

    def coords(self, y):
        th1, _, th2, _ = y
        x1, y1 = self.L1*math.sin(th1), -self.L1*math.cos(th1)
        x2, y2 = x1 + self.L2*math.sin(th2), y1 - self.L2*math.cos(th2)
        return 0, 0, x1, y1, x2, y2

    def energy(self, y):
        th1, w1, th2, w2 = y
        m1, m2, L1, L2, g = self.m1, self.m2, self.L1, self.L2, self.g
        x1, y1 = L1*math.sin(th1), -L1*math.cos(th1)
        x2, y2 = x1 + L2*math.sin(th2), y1 - L2*math.cos(th2)
        vx1, vy1 = L1*w1*math.cos(th1), L1*w1*math.sin(th1)
        vx2, vy2 = vx1 + L2*w2*math.cos(th2), vy1 + L2*w2*math.sin(th2)
        KE = 0.5*m1*(vx1**2+vy1**2)+0.5*m2*(vx2**2+vy2**2)
        PE = m1*g*(y1+L1+L2)+m2*g*(y2+L1+L2)
        return KE, PE

# Sim


class Simulator:
    def __init__(self):
        self.simple, self.double = SimplePendulum(), DoublePendulum()
        self.simple_state = np.array([math.radians(30), 0.0])
        self.double_state = np.array(
            [math.radians(120), 0.0, math.radians(-10), 0.0])
        self.system = 'Double'
        self.dt, self.t, self.running = 0.01, 0.0, False
        self.energy_hist, self.hist = [], []
        self.build_ui()

    def step(self):
        if self.system == 'Simple':
            y = rk4_step(self.simple.derivs, self.t,
                         self.simple_state, self.dt)
            self.simple_state = y
            self.t += self.dt
            KE, PE = self.simple.energy(y)
        else:
            y = rk4_step(self.double.derivs, self.t,
                         self.double_state, self.dt)
            self.double_state = y
            self.t += self.dt
            KE, PE = self.double.energy(y)
        self.energy_hist.append((self.t, KE, PE, KE+PE))
        self.hist.append(y.copy())

    def build_ui(self):
        fig = plt.figure(figsize=(9, 7))
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])
        self.ax_anim = fig.add_subplot(gs[0, 0])
        self.ax_anim.set_aspect('equal')
        self.ax_anim.set_xlim(-2.5, 2.5)
        self.ax_anim.set_ylim(-2.5, 1)
        self.ax_energy = fig.add_subplot(gs[1, 0])
        self.ax_controls = fig.add_subplot(gs[:, 1])
        self.ax_controls.axis('off')

        # Buttons
        ax_start = plt.axes([0.77, 0.8, 0.18, 0.05])
        ax_pause = plt.axes([0.77, 0.73, 0.18, 0.05])
        ax_reset = plt.axes([0.77, 0.66, 0.18, 0.05])
        self.btn_start = Button(ax_start, "Start")
        self.btn_pause = Button(ax_pause, "Pause")
        self.btn_reset = Button(ax_reset, "Reset")

        # Radio
        self.radio_ax = plt.axes([0.77, 0.5, 0.18, 0.1])
        self.radio = RadioButtons(self.radio_ax, ('Simple', 'Double'))
        self.radio.on_clicked(self._change_system)

        # Sliders
        self.sliders = {}
        sL1 = plt.axes([0.77, 0.42, 0.18, 0.03])
        sθ1 = plt.axes([0.77, 0.36, 0.18, 0.03])
        sdamp = plt.axes([0.77, 0.3, 0.18, 0.03])
        sdt = plt.axes([0.77, 0.24, 0.18, 0.03])
        self.sliders['L1'] = Slider(sL1, 'L', 0.5, 2.5, valinit=1.0)
        self.sliders['θ1'] = Slider(sθ1, 'θ1', -180, 180, valinit=120)
        self.sliders['damp'] = Slider(sdamp, 'damp', 0, 1, valinit=0)
        self.sliders['dt'] = Slider(sdt, 'dt', 0.001, 0.05, valinit=0.01)
        for s in self.sliders.values():
            s.on_changed(self._slider_update)

        # Plots
        self.line, = self.ax_anim.plot([], [], '-', lw=2)
        self.bobs, = self.ax_anim.plot([], [], 'o', markersize=10)
        self.e_line_ke, = self.ax_energy.plot([], [], label='KE')
        self.e_line_pe, = self.ax_energy.plot([], [], label='PE')
        self.e_line_tot, = self.ax_energy.plot([], [], label='Total')
        self.ax_energy.legend()

        # Events
        self.btn_start.on_clicked(lambda e: setattr(self, "running", True))
        self.btn_pause.on_clicked(lambda e: setattr(self, "running", False))
        self.btn_reset.on_clicked(self._reset)
        self.anim = FuncAnimation(fig, self._update, interval=25, blit=True)
        plt.show()

    def _change_system(self, label):
        self.system = label
        self._reset(None)

    def _slider_update(self, val):
        self.simple.L = self.sliders['L1'].val
        self.double.L1 = self.sliders['L1'].val
        self.simple.damping = self.sliders['damp'].val
        self.double.damping = self.sliders['damp'].val
        self.dt = self.sliders['dt'].val
        th = math.radians(self.sliders['θ1'].val)
        self.simple_state[0] = th
        self.double_state[0] = th
        self._draw_frame()

    def _reset(self, _):
        self.t = 0
        self.simple_state = np.array([math.radians(30), 0.0])
        self.double_state = np.array(
            [math.radians(120), 0.0, math.radians(-10), 0.0])
        self.energy_hist.clear()
        self.hist.clear()
        self._draw_frame()

    def _update(self, _):
        if self.running:
            for _ in range(max(1, int(0.02/self.dt))):
                self.step()
        return self._draw_frame()

    def _draw_frame(self):
        if self.system == 'Simple':
            x, y = self.simple.coords(self.simple_state)
            self.line.set_data([0, x], [0, y])
            self.bobs.set_data([x], [y])
        else:
            x0, y0, x1, y1, x2, y2 = self.double.coords(self.double_state)
            self.line.set_data([x0, x1, x2], [y0, y1, y2])
            self.bobs.set_data([x1, x2], [y1, y2])
        if self.energy_hist:
            t, KE, PE, TOT = zip(*self.energy_hist)
            self.e_line_ke.set_data(t, KE)
            self.e_line_pe.set_data(t, PE)
            self.e_line_tot.set_data(t, TOT)
            self.ax_energy.relim()
            self.ax_energy.autoscale_view()
        return self.line, self.bobs, self.e_line_ke, self.e_line_pe, self.e_line_tot


if __name__ == "__main__":
    Simulator()
