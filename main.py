import random
import math
import logging

import agentpy as ap
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu

# ------- Importa la clase OBJ para cargar el modelo 3D -------
from objloader import OBJ  # Asegúrate de tener objloader.py en el mismo dir

###############################################################################
#                          CONSTANTES Y COORDENADAS                           #
###############################################################################
X = [-95, -55, -20, -10, 10, 20, 55, 95]
Y = [95, 55, 20, 10, -10, -20, -55, -95]

logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
)

logging.info(
    "Iniciando simulación con dos tipos de calles (normal y luz) usando mensajes..."
)

###############################################################################
#                          CONFIGURACIÓN DEL MODELO                           #
###############################################################################
NODE_POS = {
    1: (X[7], Y[4]),
    2: (X[6], Y[4]),
    3: (X[5], Y[4]),
    4: (X[2], Y[4]),
    5: (X[0], Y[4]),
    6: (X[0], Y[7]),
    7: (X[3], Y[7]),
    8: (X[3], Y[6]),
    9: (X[3], Y[5]),
    10: (X[3], Y[2]),
    11: (X[3], Y[0]),
    12: (X[0], Y[0]),
    13: (X[0], Y[3]),
    14: (X[1], Y[3]),
    15: (X[2], Y[3]),
    16: (X[5], Y[3]),
    17: (X[7], Y[3]),
    18: (X[7], Y[0]),
    19: (X[4], Y[0]),
    20: (X[4], Y[1]),
    21: (X[4], Y[2]),
    22: (X[4], Y[5]),
    23: (X[4], Y[7]),
    24: (X[7], Y[7]),
}

EDGES = {
    1: [(2, "normal")],
    2: [(3, "luz")],
    3: [(4, "interseccion"), (10, "interseccion"), (22, "interseccion")],
    4: [(5, "normal")],
    5: [(6, "normal")],
    6: [(7, "normal")],
    7: [(8, "normal")],
    8: [(9, "luz")],
    9: [(10, "interseccion"), (4, "interseccion"), (16, "interseccion")],
    10: [(11, "normal")],
    11: [(12, "normal")],
    12: [(13, "normal")],
    13: [(14, "normal")],
    14: [(15, "luz")],
    15: [(16, "interseccion"), (10, "interseccion"), (22, "interseccion")],
    16: [(17, "normal")],
    17: [(18, "normal")],
    18: [(19, "normal")],
    19: [(20, "normal")],
    20: [(21, "luz")],
    21: [(22, "interseccion"), (16, "interseccion"), (4, "interseccion")],
    22: [(23, "normal")],
    23: [(24, "normal")],
    24: [(1, "normal")],
}


def midpoint(p1, p2):
    """Punto intermedio entre p1 y p2 (2D)."""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


# Identificamos cuáles edges tienen semáforo ("luz")
LIGHTS_INFO = []
for edge in [(2, 3), (8, 9), (14, 15), (20, 21)]:
    pm = midpoint(NODE_POS[edge[0]], NODE_POS[edge[1]])
    LIGHTS_INFO.append(
        {
            "name": f"TL{edge[0]}{edge[1]}",
            "edge": edge,
            "pos": (pm[0], 3, pm[1]),
        }
    )

GREEN_TIME = 40
YELLOW_TIME = 15
NUM_VEHICLES = 5


###############################################################################
#                         CLASES BÁSICAS (Message, Node, Graph)               #
###############################################################################
class Message:
    buffer = []

    def __init__(self, sender=None, receiver=None, perf=None, content=None):
        self.sender = sender
        self.receiver = receiver
        self.perf = perf
        self.content = content

    def send(self):
        Message.buffer.append(self)


class Node:
    """Nodo envuelto con posición (x,0,z)."""

    def __init__(self, nid, pos):
        self.id = nid
        self.pos = (pos[0], 0, pos[1])


class Graph:
    """Grafo con nodos y edges."""

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def get_node(self, node_id):
        return self.nodes[node_id]

    def next_moves(self, node_id):
        return self.edges.get(node_id, [])


nodes = {n: Node(n, p) for n, p in NODE_POS.items()}
graph = Graph(nodes, EDGES)


###############################################################################
#                        AGENTE SEMÁFORO (TrafficLightAgent)                  #
###############################################################################
class TrafficLightAgent(ap.Agent):
    """Semáforo con fases: 'red', 'green', 'yellow'."""

    def setup(self):
        self.edge = None
        self.phase = "red"
        self.busy = False
        self.green_t = GREEN_TIME
        self.yellow_t = YELLOW_TIME
        self.timer = 0
        self.pos = (0, 3, 0)

    def step(self):
        self.receive_messages()
        self.update_light()

    def receive_messages(self):
        to_remove = []
        for msg in Message.buffer:
            if msg.receiver == self.id or msg.receiver is None:
                self.handle_message(msg)
                to_remove.append(msg)
        for m in to_remove:
            Message.buffer.remove(m)

    def handle_message(self, msg):
        if msg.perf == "request_green":
            logging.debug(
                f"[TrafficLightAgent {self.id}] request_green de Vehículo {msg.sender}"
            )
            can_go_green = self.can_turn_green()
            if can_go_green and not self.busy:
                self.phase = "green"
                self.busy = True
                self.timer = self.green_t
                logging.debug(
                    f"[TrafficLightAgent {self.id}] -> GREEN (por request de Veh {msg.sender})"
                )
                # Avisamos a otros semáforos
                broadcast = Message(
                    sender=self.id,
                    receiver=None,
                    perf="turning_green",
                    content=self.id,
                )
                broadcast.send()

        elif msg.perf == "turning_green":
            if msg.content != self.id:
                if self.phase in ("green", "yellow"):
                    self.phase = "red"
                    self.busy = False
                    self.timer = 0
                    logging.debug(
                        f"[TrafficLightAgent {self.id}] -> RED (otro {msg.content} se puso verde)"
                    )

        else:
            logging.debug(
                f"[TrafficLightAgent {self.id}] Mensaje desconocido: {msg.perf}"
            )

    def can_turn_green(self):
        """Chequea si hay otro semáforo en verde o amarillo."""
        for tl in self.model.lights:
            if tl.id != self.id and (tl.phase == "green" or tl.phase == "yellow"):
                return False
        return True

    def update_light(self):
        """Transición green->yellow->red."""
        if not self.busy:
            return
        self.timer -= 1
        if self.timer <= 0:
            if self.phase == "green":
                self.phase = "yellow"
                self.timer = self.yellow_t
                logging.debug(f"[TrafficLightAgent {self.id}] -> YELLOW")
            elif self.phase == "yellow":
                self.phase = "red"
                self.busy = False
                logging.debug(f"[TrafficLightAgent {self.id}] -> RED (fin ciclo)")


###############################################################################
#                              AGENTE VEHÍCULO                                #
###############################################################################
def dynamic_deceleration(dist, max_dist=10, max_decel=25.0):
    """
    Devuelve cuánta desaceleración aplicar en base a la distancia.
      - dist>max_dist => 0
      - dist=0        => max_decel
      - lineal entre 0 y max_dist
    """
    if dist <= 0:
        return max_decel
    if dist >= max_dist:
        return 0
    factor = 1.0 - (dist / max_dist)
    return max_decel * factor


class VehicleAgent(ap.Agent):
    def setup(self):
        # Elige un edge al azar que no sea interseccion
        non_int_edges = []
        for s_node, conn_list in EDGES.items():
            for d_node, rtype in conn_list:
                if rtype != "interseccion":
                    non_int_edges.append((s_node, d_node, rtype))

        s, d, road_type = random.choice(non_int_edges)
        ps = graph.get_node(s).pos  # (x,0,z)
        pd = graph.get_node(d).pos

        t = random.random()  # progreso inicial en [0..1]
        x = ps[0] + (pd[0] - ps[0]) * t
        z = ps[2] + (pd[2] - ps[2]) * t
        y = ps[1]  # Normalmente 0

        self.position = (x, y, z)
        self.node = s
        self.current_road_type = road_type
        self.current_edge = (s, d)

        self.start_pos = (ps[0], y, ps[2])
        self.end_pos = (pd[0], y, pd[2])
        dx = pd[0] - ps[0]
        dz = pd[2] - ps[2]
        self.dist = math.hypot(dx, dz)
        self.prog = t

        self.speed = 1
        self.max_speed = 5
        self.acc = 0.5
        self.finished = False

        # (IMPORTANTE) Heading inicial: calculamos el ángulo desde (dx, dz)
        angle_radians = math.atan2(dz, dx)
        self.heading = math.degrees(angle_radians)

        logging.debug(
            f"[VehicleAgent #{self.id}] edge=({s}->{d}), t={t:.2f}, heading={self.heading:.1f}"
        )

    def step(self):
        if self.finished:
            return

        self.receive_messages()

        dec_total = 0.0
        dec_total += self.sense_vehicle_ahead()
        dec_total += self.handle_traffic_light_if_needed()

        if dec_total > 0:
            new_speed = self.speed - dec_total
            self.speed = max(0, new_speed)
            logging.debug(
                f"[Vehicle #{self.id}] BRAKE -> speed={self.speed:.2f} (dec={dec_total:.2f})"
            )
        else:
            old_speed = self.speed
            self.speed = min(self.speed + self.acc, self.max_speed)
            if self.speed > old_speed:
                logging.debug(f"[Vehicle #{self.id}] ACCEL -> speed={self.speed:.2f}")

        self.move()

    def receive_messages(self):
        to_remove = []
        for msg in Message.buffer:
            if msg.receiver == self.id:
                logging.debug(f"[Vehicle #{self.id}] Mensaje recibido: {msg.perf}")
                to_remove.append(msg)
        for m in to_remove:
            Message.buffer.remove(m)

    def sense_vehicle_ahead(self):
        if self.finished or self.prog >= 1:
            return 0
        dir_x = self.end_pos[0] - self.position[0]
        dir_z = self.end_pos[2] - self.position[2]
        length = math.hypot(dir_x, dir_z)
        if length < 1e-9:
            return 0
        dir_x /= length
        dir_z /= length

        min_d = None
        for o in self.model.vehicles:
            if o.id == self.id or o.finished:
                continue
            ox = o.position[0] - self.position[0]
            oz = o.position[2] - self.position[2]
            dot = (dir_x * ox) + (dir_z * oz)
            if dot > 0:  # Delante
                dist_veh = math.hypot(ox, oz)
                if dist_veh < 10:
                    if (min_d is None) or (dist_veh < min_d):
                        min_d = dist_veh
        if min_d is not None:
            return dynamic_deceleration(min_d, max_dist=10, max_decel=20.0)
        return 0.0

    def handle_traffic_light_if_needed(self):
        if self.current_road_type != "luz" or not self.current_edge:
            return 0.0
        for tl in self.model.lights:
            if tl.edge == self.current_edge:
                msg = Message(
                    sender=self.id,
                    receiver=tl.id,
                    perf="request_green",
                    content={"edge": self.current_edge},
                )
                msg.send()
                if tl.phase == "red":
                    dist_to_end = self.dist * (1 - self.prog)
                    dec = dynamic_deceleration(
                        dist_to_end, max_dist=self.dist, max_decel=20.0
                    )
                    logging.debug(
                        f"[Vehicle #{self.id}] TL {tl.id} ROJO. dist={dist_to_end:.1f}, dec={dec:.2f}"
                    )
                    return dec
                return 0.0
        return 0.0

    def move(self):
        if self.finished:
            return

        distance_step = self.speed
        still_to_move = distance_step

        while still_to_move > 1e-9 and not self.finished:
            dist_remaining = self.dist * (1 - self.prog)
            if still_to_move >= dist_remaining:
                self.prog = 1
                self.position = (self.end_pos[0], self.start_pos[1], self.end_pos[2])
                self.node = self.find_node(self.position)
                still_to_move -= dist_remaining

                moves = graph.next_moves(self.node)
                if not moves:
                    self.finished = True
                    break

                (dest_node, road_label) = random.choice(moves)
                self.current_road_type = road_label
                self.current_edge = (self.node, dest_node)

                self.start_pos = self.position
                self.end_pos = graph.get_node(dest_node).pos

                dx = self.end_pos[0] - self.start_pos[0]
                dz = self.end_pos[2] - self.start_pos[2]
                self.dist = math.hypot(dx, dz)
                self.prog = 0

                # (IMPORTANTE) Recalcular heading
                angle_radians = math.atan2(dz, dx)
                self.heading = math.degrees(angle_radians)
                logging.debug(f"[Vehicle #{self.id}] Nuevo heading={self.heading:.1f}")

            else:
                delta_prog = still_to_move / (self.dist + 1e-9)
                self.prog += delta_prog
                still_to_move = 0

            if self.prog < 1:
                x = (
                    self.start_pos[0]
                    + (self.end_pos[0] - self.start_pos[0]) * self.prog
                )
                z = (
                    self.start_pos[2]
                    + (self.end_pos[2] - self.start_pos[2]) * self.prog
                )
                self.position = (x, self.start_pos[1], z)

    def find_node(self, p):
        for name, nd in graph.nodes.items():
            if abs(nd.pos[0] - p[0]) < 1e-3 and abs(nd.pos[2] - p[2]) < 1e-3:
                return name
        return None


###############################################################################
#                              MODELO PRINCIPAL                               #
###############################################################################
class TwoTypeStreetsModel(ap.Model):
    def setup(self):
        logging.info("TwoTypeStreetsModel.setup()")
        self.lights = ap.AgentList(self, len(LIGHTS_INFO), TrafficLightAgent)
        for i, tl in enumerate(self.lights):
            tl.edge = LIGHTS_INFO[i]["edge"]
            tl.pos = LIGHTS_INFO[i]["pos"]
            logging.debug(f"TrafficLightAgent {tl.id} edge={tl.edge}, pos={tl.pos}")

        self.vehicles = ap.AgentList(self, NUM_VEHICLES, VehicleAgent)

    def step(self):
        logging.debug("===== TwoTypeStreetsModel step() =====")
        self.lights.step()
        self.vehicles.step()


###############################################################################
#                           VISUALIZACIÓN OPENGL                              #
###############################################################################
car_model = None  # Se cargará en main()


def draw_axes():
    gl.glBegin(gl.GL_LINES)
    # Eje X
    gl.glColor3f(1, 0, 0)
    gl.glVertex3f(-50, 0, 0)
    gl.glVertex3f(50, 0, 0)
    # Eje Y
    gl.glColor3f(0, 1, 0)
    gl.glVertex3f(0, -50, 0)
    gl.glVertex3f(0, 50, 0)
    # Eje Z
    gl.glColor3f(0, 0, 1)
    gl.glVertex3f(0, 0, -50)
    gl.glVertex3f(0, 0, 50)
    gl.glEnd()


def draw_lines():
    # Líneas amarillas, etc.
    rectangles = [
        ((-2, -85), (-1, -30)),
        ((1, -85), (2, -30)),
        ((-2, 30), (-1, 85)),
        ((1, 30), (2, 85)),
        ((-85, -2), (-30, -1)),
        ((-85, 2), (-30, 1)),
        ((30, -2), (85, -1)),
        ((30, 2), (85, 1)),
        ((-85, -2), (-30, -1)),
        ((-85, 2), (-30, 1)),
        ((30, -2), (85, -1)),
        ((30, 2), (85, 1)),
        ((-2, 85), (-1, 30)),
        ((1, 85), (2, 30)),
        ((-2, -30), (-1, -85)),
        ((1, -30), (2, -85)),
    ]
    gl.glColor3f(1, 1, 0)
    for r in rectangles:
        x1, z1 = r[0]
        x2, z2 = r[1]
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex3f(x1, 1, z1)
        gl.glVertex3f(x1, 1, z2)
        gl.glVertex3f(x2, 1, z2)
        gl.glVertex3f(x2, 1, z1)
        gl.glEnd()


def draw_roads():
    for start_node, connections in EDGES.items():
        p1 = graph.get_node(start_node).pos
        x1, _, z1 = p1
        for dest_node, label in connections:
            p2 = graph.get_node(dest_node).pos
            x2, _, z2 = p2
            dx = x2 - x1
            dz = z2 - z1
            length = math.hypot(dx, dz)
            if length < 1e-9:
                continue

            ndx = dx / length
            ndz = dz / length
            # perpendicular
            px = -ndz
            pz = ndx
            offset = 10.0
            ox = px * offset
            oz = pz * offset

            if label in ("normal", "luz"):
                p1_left = (x1 - ox, 0, z1 - oz)
                p1_right = (x1 + ox, 0, z1 + oz)
                p2_left = (x2 - ox, 0, z2 - oz)
                p2_right = (x2 + ox, 0, z2 + oz)

                gl.glColor3f(0.3, 0.3, 0.3)
                gl.glBegin(gl.GL_QUADS)
                gl.glVertex3f(*p1_left)
                gl.glVertex3f(*p1_right)
                gl.glVertex3f(*p2_right)
                gl.glVertex3f(*p2_left)
                gl.glEnd()
            else:
                # interseccion -> línea punteada
                gl.glColor3f(1, 1, 1)
                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(x1, 1, z1)
                gl.glVertex3f(x2, 1, z2)
                gl.glEnd()

# Posiciones de los árboles y edificios en las áreas verdes
tree_positions = [
    (-32, 0, -35),# Ejemplo de posición para un árbol
    (-32, 0, -70),
    (-32, 0, 35),   # Otra posición para un árbol
    (-32, 0, 70),
    (32, 0, -35),   # Otra posición para un árbol
    (32, 0, -70),
    (32, 0, 35),     # Otra posición para un árbol
    (32, 0, 70)
]

building_positions = [
    ((-45, 0, -52), -90),  # Rotar 90° para alinearlo con la calle
    ((-45, 0, 52), -90),   # No rotar
    ((45, 0, -52), 90),  # Rotar 90° para alinearlo con la calle
    ((45, 0, 52), 90)    # No rotar
]

house_positions = [
    ((-62, 0, -52),90),  # Ejemplo de posición para una casa
    ((-62, 0, -30),90),
    ((-62, 0, 52), 90),
    ((-62, 0, 75), 90),
    ((62, 0, -75), -90),
    ((62, 0, -47), -90),
    ((62, 0, 57),-90),
    ((62, 0, 35),-90),
]

walls = [
    ((-100, 0, -100), (100, 50, -100)),  # Muro trasero
    ((-100, 0, 100), (100, 50, 100)),    # Muro frontal
    ((-100, 0, -100), (-100, 50, 100)),  # Muro izquierdo
    ((100, 0, -100), (100, 50, 100)),    # Muro derecho
]


def draw_tree(position):
    x, y, z = position
    gl.glPushMatrix()
    gl.glTranslatef(x, y, z)
    gl.glRotatef(-90, 1, 0, 0)
    gl.glScalef(1, 1, 1)  # Ajusta la escala del árbol si es necesario
    tree_model.render()
    gl.glPopMatrix()

def draw_building(position, rotation):
    x, y, z = position
    gl.glPushMatrix()
    gl.glTranslatef(x, y, z)  # Mueve el edificio a su posición
    gl.glRotatef(rotation, 0, 1, 0)  # Gira según el ángulo dado
    gl.glRotatef(-90, 1, 0, 0)
    gl.glScalef(1.1, 1.1, 1.1)  # Ajusta la escala del edificio
    
    # Activar textura
    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glBindTexture(gl.GL_TEXTURE_2D, building_texture)

    # Renderizar edificio
    building_model.render()

    # Desactivar textura
    gl.glDisable(gl.GL_TEXTURE_2D)

    gl.glPopMatrix()

def draw_walls():
    gl.glEnable(gl.GL_TEXTURE_2D)  # Usa el prefijo gl.
    gl.glBindTexture(gl.GL_TEXTURE_2D, wall_texture)

    gl.glColor3f(1, 1, 1)  # Asegura que la textura no se vea afectada por el color

    for p1, p2 in walls:
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 0); gl.glVertex3f(x1, y1, z1)
        gl.glTexCoord2f(1, 0); gl.glVertex3f(x2, y1, z2)
        gl.glTexCoord2f(1, 1); gl.glVertex3f(x2, y2, z2)
        gl.glTexCoord2f(0, 1); gl.glVertex3f(x1, y2, z1)
        gl.glEnd()

    gl.glDisable(gl.GL_TEXTURE_2D)  # Usa el prefijo gl.

def draw_house(position, rotation):
    x, y, z = position
    gl.glPushMatrix()
    gl.glTranslatef(x, y, z)
    gl.glRotatef(rotation, 0, 1, 0)  # Ajusta la rotación si es necesario
    gl.glRotatef(-90, 1, 0, 0)
    gl.glScalef(12, 12, 12)  # Ajusta la escala de la casa si es necesario
    house_model.render()
    gl.glPopMatrix()


def draw_areas():
    # Define quads using only two opposite points
    quads_gray = [
        ((X[2], -0.1, X[2]), (X[5], -0.1, X[5])),
        ((-85, -0.1, -85), (X[2], -0.1, 85)),
        ((-85, -0.1, X[5]), (X[2], -0.1, 85)),
        ((X[5], -0.1, -85), (85, -0.1, X[2])),
        ((X[5], -0.1, X[5]), (85, -0.1, 85)),
    ]

    quads_green = [
        ((-75, 0, -75), (X[2] - 10, 0, X[2] - 10)),
        ((-75, 0, X[5] + 10), (X[2] - 10, 0, 75)),
        ((X[5] + 10, 0, -75), (75, 0, X[2] - 10)),
        ((X[5] + 10, 0, X[5] + 10), (75, 0, 75)),
    ]
    
    def get_quad_points(p1, p2):
        """Generate the four corners of a quad from two opposite corners."""
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return [(x1, y1, z1), (x2, y1, z1), (x2, y1, z2), (x1, y1, z2)]

    def draw_quads(quads, color):
        gl.glColor3f(*color)
        for p1, p2 in quads:
            quad = get_quad_points(p1, p2)
            gl.glBegin(gl.GL_QUADS)
            for v in quad:
                gl.glVertex3f(*v)
            gl.glEnd()

    draw_quads(quads_gray, (0.6, 0.6, 0.6))
    draw_quads(quads_green, (0.2, 0.8, 0.2))


def draw_crosswalk(p1, p2, width=10, num_stripes=8):
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    x1, y1, z1 = p1
    x2, y2, z2 = p2
    dx = x2 - x1
    dz = z2 - z1
    length = math.hypot(dx, dz)
    if length < 1e-9:
        gl.glDisable(gl.GL_BLEND)
        return

    ndx = dx / length
    ndz = dz / length
    px = ndz
    pz = -ndx
    half_w = width / 2.0

    p1_left = (x1 - half_w * px, y1, z1 - half_w * pz)
    p1_right = (x1 + half_w * px, y1, z1 + half_w * pz)
    p2_left = (x2 - half_w * px, y2, z2 - half_w * pz)
    p2_right = (x2 + half_w * px, y2, z2 + half_w * pz)

    stripe_len = length / num_stripes
    traveled = 0.0
    for i in range(num_stripes):
        start_dist = traveled
        end_dist = traveled + stripe_len
        s1 = start_dist / length
        s2 = end_dist / length

        # Interpolamos
        f1_left_x = p1_left[0] + (p2_left[0] - p1_left[0]) * s1
        f1_left_z = p1_left[2] + (p2_left[2] - p1_left[2]) * s1
        f1_right_x = p1_right[0] + (p2_right[0] - p1_right[0]) * s1
        f1_right_z = p1_right[2] + (p2_right[2] - p1_right[2]) * s1

        f2_left_x = p1_left[0] + (p2_left[0] - p1_left[0]) * s2
        f2_left_z = p1_left[2] + (p2_left[2] - p1_left[2]) * s2
        f2_right_x = p1_right[0] + (p2_right[0] - p1_right[0]) * s2
        f2_right_z = p1_right[2] + (p2_right[2] - p1_right[2]) * s2

        if i % 2 == 0:
            gl.glColor4f(1, 1, 1, 1)
        else:
            gl.glColor4f(1, 1, 1, 0)

        gl.glBegin(gl.GL_QUADS)
        gl.glVertex3f(f1_left_x, 0.1, f1_left_z)
        gl.glVertex3f(f1_right_x, 0.1, f1_right_z)
        gl.glVertex3f(f2_right_x, 0.1, f2_right_z)
        gl.glVertex3f(f2_left_x, 0.1, f2_left_z)
        gl.glEnd()

        traveled += stripe_len
    

    gl.glDisable(gl.GL_BLEND)


def draw_light(tl):
    x, y, z = tl.pos
    gl.glPushMatrix()
    gl.glTranslatef(x, y, z)
    if tl.phase == "green":
        gl.glColor3f(0, 1, 0)
    elif tl.phase == "yellow":
        gl.glColor3f(1, 1, 0)
    else:
        gl.glColor3f(1, 0, 0)
    # 3 esferas
    for offset in [5, 0, -5]:
        gl.glPushMatrix()
        gl.glTranslatef(0, offset, 0)
        glut.glutSolidSphere(1, 16, 16)
        gl.glPopMatrix()
    gl.glPopMatrix()


def draw_vehicle(vehicle):
    """
    (IMPORTANTE) Aquí usamos vehicle.heading para rotar el coche.
    """
    if vehicle.finished:
        return
    px, py, pz = vehicle.position

    gl.glPushMatrix()
    gl.glTranslatef(px, py, pz)

    # Si tu modelo OBJ está 'acostado', ajusta la rotación base
    gl.glRotatef(-90, 1, 0, 0)  # Ejemplo, si tu .obj apunta en eje +Z...
    gl.glRotatef(-90, 0, 0, 1)  # Ejemplo, si tu .obj apunta en eje +Z...
    # Rotamos según heading (sobre Y)
    gl.glRotatef(-vehicle.heading, 0, 0, 1)

    # Escalamos
    gl.glScalef(1.5, 1.5, 1.5)

    # Render del modelo
    car_model.render()

    gl.glPopMatrix()


def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    glu.gluLookAt(0, 150, 150, 0, 0, 0, 0, 1, 0)

    draw_areas()
    draw_axes()
    draw_lines()
    draw_roads()
    draw_walls()
    
    # Dibujar árboles en las áreas verdes
    for pos in tree_positions:
        draw_tree(pos)

    # Dibujar edificios en las áreas verdes
    for pos, rot in building_positions:
       draw_building(pos, rot)
       
    # Dibujar casas en las áreas verdes
    for pos, rot in house_positions:
        draw_house(pos, rot)

    
    crosswalk_lines = [
        ((25, 0, -20), (25, 0, 20)),
        ((-25, 0, -20), (-25, 0, 20)),
        ((-20, 0, 25), (20, 0, 25)),
        ((-20, 0, -25), (20, 0, -25)),
    ]
    for p1, p2 in crosswalk_lines:
        draw_crosswalk(p1, p2)

    # Vehículos
    for v in model.vehicles:
        draw_vehicle(v)

    # Semáforos
    for tl in model.lights:
        draw_light(tl)

    glut.glutSwapBuffers()


def update(_):
    model.step()
    glut.glutPostRedisplay()
    glut.glutTimerFunc(100, update, 0)


def main():
    logging.info(
        "main() -> Iniciando simulación con mensajes y orientaciones de vehículo."
    )
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB | glut.GLUT_DEPTH)
    glut.glutInitWindowSize(800, 600)
    glut.glutCreateWindow(b"Calles normales e iluminadas - Mensajes + Heading")

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glClearColor(0, 0, 0, 1)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(60, 800 / 600, 1, 1000)

    glut.glutDisplayFunc(display)
    glut.glutTimerFunc(100, update, 0)

    # Cargar modelo .obj (carro)
    global car_model
    car_model = OBJ("Assets/Chevrolet_Camaro_SS_Low.obj", swapyz=True)  # Ajusta el nombre/ruta
    car_model.generate()
    
    # Cargar modelo .obj (árbol)
    global tree_model
    tree_model = OBJ("Assets/CartoonTree.obj", swapyz=True)  # Ajusta el nombre/ruta
    tree_model.generate()

    # Cargar modelo .obj (edificio)
    global building_model
    building_model = OBJ("Assets/Rv_Building_3.obj", swapyz=True)  # Ajusta el nombre/ruta
    building_model.generate()
    
    global building_texture
    building_texture = OBJ.loadTexture("Assets/texture_build.jpg")  # Carga la textura de ladrillos

    # Cargar modelo .obj (casa)
    global house_model
    house_model = OBJ("Assets/House/House.obj", swapyz=True)  # Ajusta el nombre/ruta
    house_model.generate()
    
    global wall_texture
    wall_texture = OBJ.loadTexture("Assets/background.jpg")  # Cargar la textura de la imagen


    global model
    model = TwoTypeStreetsModel({})
    model.setup()

    logging.info("main() -> Entrando a glutMainLoop.")
    glut.glutMainLoop()


if __name__ == "__main__":
    main()
