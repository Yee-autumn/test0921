""" A* + DWA，简易静态地图，动态障碍物固定y=20，x在[-10,10]左右往复移动 """
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['figure.autolayout'] = False
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

show_animation = True
show_search = False

# ===================== AStarPlanner =====================
class AStarPlanner:
    def __init__(self, ox, oy, resolution, rr):
        self.resolution = resolution
        self.rr = rr
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

    def planning(self, sx, sy, gx, gy, ax=None):
        nstart = self.Node(self.calc_xyindex(sx, self.minx),
                           self.calc_xyindex(sy, self.miny), 0.0, -1)
        ngoal = self.Node(self.calc_xyindex(gx, self.minx),
                          self.calc_xyindex(gy, self.miny), 0.0, -1)
        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(nstart)] = nstart

        while 1:
            if len(open_set) == 0:
                print("Open_set is empty...")
                break
            c_id = min(open_set,
                       key=lambda o: open_set[o].cost + self.calc_heuristic(ngoal, open_set[o]))
            current = open_set[c_id]

            if show_search and ax is not None:
                ax.plot(self.calc_grid_position(current.x, self.minx),
                        self.calc_grid_position(current.y, self.miny), "xc")
                if len(closed_set.keys()) % 20 == 0:
                    ax.figure.canvas.flush_events()

            if current.x == ngoal.x and current.y == ngoal.y:
                print("Find goal!")
                ngoal.parent_index = current.parent_index
                ngoal.cost = current.cost
                break

            del open_set[c_id]
            closed_set[c_id] = current

            for move_x, move_y, move_cost in self.motion:
                node = self.Node(current.x + move_x,
                                 current.y + move_y,
                                 current.cost + move_cost, c_id)
                n_id = self.calc_grid_index(node)
                if not self.verify_node(node):
                    continue
                if n_id in closed_set:
                    continue
                if n_id not in open_set:
                    open_set[n_id] = node
                elif open_set[n_id].cost > node.cost:
                    open_set[n_id] = node

        pathx, pathy = self.calc_final_path(ngoal, closed_set)
        return pathx, pathy

    def calc_final_path(self, ngoal, closedset):
        pathx, pathy = [self.calc_grid_position(ngoal.x, self.minx)], [
            self.calc_grid_position(ngoal.y, self.miny)]
        parent_index = ngoal.parent_index
        while parent_index != -1:
            n = closedset[parent_index]
            pathx.append(self.calc_grid_position(n.x, self.minx))
            pathy.append(self.calc_grid_position(n.y, self.miny))
            parent_index = n.parent_index
        return pathx, pathy

    @staticmethod
    def calc_heuristic(n1, n2):
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def calc_grid_position(self, index, minpos):
        return index * self.resolution + minpos

    def calc_xyindex(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.miny) * self.xwidth + (node.x - self.minx)

    def verify_node(self, node):
        posx = self.calc_grid_position(node.x, self.minx)
        posy = self.calc_grid_position(node.y, self.miny)
        if posx < self.minx or posy < self.miny:
            return False
        if posx >= self.maxx or posy >= self.maxy:
            return False
        return not self.obmap[int(node.x)][int(node.y)]

    def calc_obstacle_map(self, ox, oy):
        self.minx = round(min(ox))
        self.miny = round(min(oy))
        self.maxx = round(max(ox))
        self.maxy = round(max(oy))
        self.xwidth = round((self.maxx - self.minx) / self.resolution)
        self.ywidth = round((self.maxy - self.miny) / self.resolution)

        gx = self.calc_grid_position(np.arange(self.xwidth), self.minx)
        gy = self.calc_grid_position(np.arange(self.ywidth), self.miny)
        grid_x, grid_y = np.meshgrid(gx, gy, indexing='ij')
        ox = np.asarray(ox); oy = np.asarray(oy)
        d2 = (grid_x[..., None] - ox)**2 + (grid_y[..., None] - oy)**2
        min_d = np.sqrt(d2.min(axis=-1))
        self.obmap = min_d <= self.rr

    @staticmethod
    def get_motion_model():
        return [[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
                [1, 1, math.sqrt(2)], [1, -1, math.sqrt(2)],
                [-1, 1, math.sqrt(2)], [-1, -1, math.sqrt(2)]]

# ===================== DWA 局部规划 =====================
class DWA:
    def __init__(self, ox, oy, robot_radius):
        self.max_speed = 2.0
        self.min_speed = -0.5
        self.max_yaw_rate = math.pi / 2
        self.max_accel = 0.5
        self.max_dyaw_acc = math.pi / 4

        self.dt = 0.1
        self.predict_time = 3.0
        self.n_steps = int(self.predict_time / self.dt) + 1

        self.to_goal_cost_gain = 0.2
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.2
        self.robot_radius = robot_radius

    def plan(self, x, goal, ax, candidate_lines, static_ox, static_oy, dyn_obs):
        dw = self.calc_dynamic_window(x)
        all_ox = static_ox + [dyn_obs[0]]
        all_oy = static_oy + [dyn_obs[1]]
        self.obs_tree = cKDTree(np.column_stack([all_ox, all_oy]))
        u, trajectory = self.calc_control_and_trajectory(x, dw, goal, ax, candidate_lines)
        return u[0], u[1], trajectory

    def calc_dynamic_window(self, x):
        return [
            max(self.min_speed, x[3] - self.max_accel * self.dt),
            min(self.max_speed, x[3] + self.max_accel * self.dt),
            max(-self.max_yaw_rate, x[4] - self.max_dyaw_acc * self.dt),
            min(self.max_yaw_rate, x[4] + self.max_dyaw_acc * self.dt),
        ]

    def calc_control_and_trajectory(self, x, dw, goal, ax, candidate_lines):
        best_cost = float("inf")
        best_u = [0.0, 0.0]
        best_traj = np.empty((self.n_steps, 5))
        best_traj[:] = x
        traj = np.empty((self.n_steps, 5))

        for l in candidate_lines:
            l.remove()
        candidate_lines.clear()

        for v in np.arange(dw[0], dw[1], 0.1):
            for w in np.arange(dw[2], dw[3], 0.05):
                self.predict_trajectory(x, v, w, traj)
                d, _ = self.obs_tree.query(traj[:, :2])
                min_dist = d.min()
                if min_dist <= self.robot_radius:
                    continue
                ob_cost = 1.0 / min_dist
                goal_cost = math.hypot(goal[0] - traj[-1, 0], goal[1] - traj[-1, 1])
                speed_cost = self.speed_cost_gain * (self.max_speed - v)
                total = self.to_goal_cost_gain * goal_cost + speed_cost + self.obstacle_cost_gain * ob_cost
                if total < best_cost:
                    best_cost = total
                    best_u = [v, w]
                    best_traj[:] = traj
                line, = ax.plot(traj[:,0], traj[:,1], "-", color="gray", alpha=0.3, linewidth=1)
                candidate_lines.append(line)
        return best_u, best_traj

    def predict_trajectory(self, x, v, w, traj):
        traj[0] = x
        for i in range(1, self.n_steps):
            traj[i, 0] = traj[i-1, 0] + v * math.cos(traj[i-1, 2]) * self.dt
            traj[i, 1] = traj[i-1, 1] + v * math.sin(traj[i-1, 2]) * self.dt
            traj[i, 2] = traj[i-1, 2] + w * self.dt
            traj[i, 3] = v
            traj[i, 4] = w
        return traj

    def motion(self, x, v, w):
        x[0] += v * math.cos(x[2]) * self.dt
        x[1] += v * math.sin(x[2]) * self.dt
        x[2] += w * self.dt
        x[3] = v
        x[4] = w
        return x


def find_nearest_waypoint(robot_x, robot_y, pathx, pathy, ahead_num=3):
    d = np.hypot(np.asarray(pathx) - robot_x, np.asarray(pathy) - robot_y)
    nearest_idx = int(np.argmin(d))
    target_idx = min(nearest_idx + ahead_num, len(pathx) - 1)
    return pathx[target_idx], pathy[target_idx]

# ===================== 主函数 =====================
def main():
    print("A* + DWA simple demo")
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("A*+DWA, dynamic obstacle: y=20, x∈[-10,10]")

    sx, sy = -15.0, 0.0    # 起点
    gx, gy = 15.0, 30.0    # 终点
    grid_size = 1.0
    robot_radius = 1.0
    steps_per_frame = 2

    # ===== 简易静态地图：只有外边界，没有多余内墙 =====
    ox, oy = [], []
    # 边界框
    for i in range(-20, 21):
        ox.append(i); oy.append(-5)
    for i in range(-20, 21):
        ox.append(i); oy.append(35)
    for i in range(-5, 36):
        ox.append(-20); oy.append(i)
    for i in range(-5, 36):
        ox.append(20); oy.append(i)

    ax.plot(ox, oy, ".k")
    ax.plot(sx, sy, "og", markersize=8, label="start")
    ax.plot(gx, gy, "xb", markersize=8, label="goal")
    ax.grid(True)
    ax.axis('equal')

    # A*全局规划
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    pathx, pathy = a_star.planning(sx, sy, gx, gy, ax=ax)
    pathx = pathx[::-1]
    pathy = pathy[::-1]
    ax.plot(pathx, pathy, "-r", linewidth=2, label="A* global path")

    fig.canvas.mpl_connect('key_release_event', lambda e: exit(0) if e.key == 'escape' else None)

    robot_dot, = ax.plot([sx], [sy], "ob", markersize=7)
    pred_line, = ax.plot([sx], [sy], "y", alpha=1, linewidth=2)
    goal_mark, = ax.plot([sx], [sy], "xg")
    real_traj_line, = ax.plot([sx], [sy], "-k", linewidth=2, label="robot path")

    # ========= 动态障碍物：固定 y=20，x在[-10,10]之间左右移动 =========
    dyn_obs_x = -10.0
    dyn_obs_y = 20.0
    dyn_vx = 0.2
    dyn_obs_dot, = ax.plot([dyn_obs_x], [dyn_obs_y], "or", markersize=8, label="dynamic obstacle")

    plt.legend(loc='upper left', fontsize=8)
    fig.canvas.draw()
    candidate_lines = []

    dwa = DWA(ox, oy, robot_radius)
    x = np.array([sx, sy, 0.0, 0.0, 0.0])
    goal_dist_thresh = 1.5
    traj = np.array([[x[0], x[1], 0, 0, 0]])
    g_local_x, g_local_y = sx, sy
    real_x_history = [x[0]]
    real_y_history = [x[1]]

    while True:
        # 动态障碍物更新逻辑：y固定=20，x在[-10, 10]来回走
        dyn_obs_x += dyn_vx
        if dyn_obs_x > 10:
            dyn_vx = -0.2
        if dyn_obs_x < -10:
            dyn_vx = 0.2

        for _ in range(steps_per_frame):
            g_local_x, g_local_y = find_nearest_waypoint(x[0], x[1], pathx, pathy, ahead_num=3)
            if math.hypot(gx - x[0], gy - x[1]) < goal_dist_thresh:
                print("Reach goal!")
                break
            v, w, traj = dwa.plan(x, [g_local_x, g_local_y], ax, candidate_lines, ox, oy, [dyn_obs_x, dyn_obs_y])
            x = dwa.motion(x, v, w)
            real_x_history.append(x[0])
            real_y_history.append(x[1])

        if math.hypot(gx - x[0], gy - x[1]) < goal_dist_thresh:
            break

        if show_animation:
            robot_dot.set_data([x[0]], [x[1]])
            pred_line.set_data(traj[:, 0], traj[:, 1])
            goal_mark.set_data([g_local_x], [g_local_y])
            real_traj_line.set_data(real_x_history, real_y_history)
            dyn_obs_dot.set_data([dyn_obs_x], [dyn_obs_y])

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
