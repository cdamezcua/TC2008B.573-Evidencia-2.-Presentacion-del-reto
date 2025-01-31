import random
import math
import logging

import agentpy as ap
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu

###############################################################################
#                          NUEVAS CONSTANTES PARA COORDENADAS                 #
###############################################################################
X = [-95, -55, -20, -10, 10, 20, 55, 95]
Y = [95, 55, 20, 10, -10, -20, -55, -95]

###############################################################################
#                       CONFIGURACIÓN DE LOGGING                              #
###############################################################################
logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logging.info(
    "Iniciando simulación con dos tipos de calles (normal y luz) usando mensajes..."
)

###############################################################################
#                           CONFIGURACIÓN DEL MODELO                          #
###############################################################################
# Reemplazamos las coordenadas usando X[...] y Y[...] en lugar de valores crudos
NODE_POS = {
    1: (X[7], Y[4]),  # ( 95,   5)
    2: (X[6], Y[4]),  # ( 55,   5)
    3: (X[5], Y[4]),  # ( 15,   5)
    4: (X[2], Y[4]),  # (-15,   5)
    5: (X[0], Y[4]),  # (-95,   5)
    6: (X[0], Y[7]),  # (-95,  95)
    7: (X[3], Y[7]),  # ( -5,  95)
    8: (X[3], Y[6]),  # ( -5,  55)
    9: (X[3], Y[5]),  # ( -5,  15)
    10: (X[3], Y[2]),  # ( -5, -15)
    11: (X[3], Y[0]),  # ( -5, -95)
    12: (X[0], Y[0]),  # (-95, -95)
    13: (X[0], Y[3]),  # (-95,  -5)
    14: (X[1], Y[3]),  # (-55,  -5)
    15: (X[2], Y[3]),  # (-15,  -5)
    16: (X[5], Y[3]),  # ( 15,  -5)
    17: (X[7], Y[3]),  # ( 95,  -5)
    18: (X[7], Y[0]),  # ( 95, -95)
    19: (X[4], Y[0]),  # (  5, -95)
    20: (X[4], Y[1]),  # (  5, -55)
    21: (X[4], Y[2]),  # (  5, -15)
    22: (X[4], Y[5]),  # (  5,  15)
    23: (X[4], Y[7]),  # (  5,  95)
    24: (X[7], Y[7]),  # ( 95,  95)
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
    p_mid = midpoint(NODE_POS[edge[0]], NODE_POS[edge[1]])
    LIGHTS_INFO.append(
        {
            "name": f"TL{edge[0]}{edge[1]}",
            "edge": edge,
            "pos": (p_mid[0], 3, p_mid[1]),
        }
    )

GREEN_TIME = 40
YELLOW_TIME = 15
NUM_VEHICLES = 5

###############################################################################
#                         CLASES BÁSICAS (Message, Node, Graph)               #
###############################################################################


class Message:
    """
    Modelo sencillo de paso de mensajes.
    Usaremos esta clase para enviar y recibir mensajes entre agentes.
    """

    buffer = []  # Buffer estático y global para almacenar mensajes

    def __init__(self, sender=None, receiver=None, perf=None, content=None):
        """
        sender  : id del agente emisor
        receiver: id del agente receptor (o None para broadcast)
        perf    : cadena que indica el "acto comunicativo" (ej: "request_green")
        content : contenido adicional (dict, string, etc.)
        """
        self.sender = sender
        self.receiver = receiver
        self.perf = perf
        self.content = content

    def send(self):
        """Envía el mensaje añadiéndolo al buffer global."""
        Message.buffer.append(self)


class Node:
    """Envuelve el nombre(id) del nodo y su posición 3D (x,0,z)."""

    def __init__(self, nid, pos):
        self.id = nid
        # Guardamos en formato (x, 0, z) para 3D
        self.pos = (pos[0], 0, pos[1])


class Graph:
    """Estructura de grafos con nodos y edges para la simulación."""

    def __init__(self, nodes, edges):
        self.nodes = nodes  # dict: nodo_id -> Node
        self.edges = edges  # dict: nodo_id -> [(dest, label), ...]

    def get_node(self, node_id):
        return self.nodes[node_id]

    def next_moves(self, node_id):
        """Devuelve la lista de (dest_node_id, tipo_calle) desde node_id."""
        return self.edges.get(node_id, [])


nodes = {n: Node(n, p) for n, p in NODE_POS.items()}
graph = Graph(nodes, EDGES)

###############################################################################
#                           AGENTE SEMÁFORO (TrafficLightAgent)               #
###############################################################################


class TrafficLightAgent(ap.Agent):
    """
    Simplificamos la lógica de los 3 focos a un solo estado (phase)
    que puede ser "red", "green" o "yellow" aplicable a todos a la vez.
    """

    def setup(self):
        self.edge = None  # (origen, destino)
        self.phase = "red"  # "red", "green", "yellow"
        self.busy = False  # Indica si está en ciclo verde/amarillo
        self.green_t = GREEN_TIME
        self.yellow_t = YELLOW_TIME
        self.timer = 0

        # Posición en 3D
        self.pos = (0, 3, 0)

    def step(self):
        # 1. Procesar todos los mensajes dirigidos a este semáforo
        self.receive_messages()
        # 2. Actualizar la lógica de semáforo (transición green->yellow->red)
        self.update_light()

    def receive_messages(self):
        """Revisa el buffer global y procesa los mensajes cuyo receptor sea este semáforo."""
        to_remove = []
        for msg in Message.buffer:
            # (Si receiver es None, podría interpretarse broadcast)
            if msg.receiver == self.id or msg.receiver is None:
                self.handle_message(msg)
                to_remove.append(msg)
        # Limpiamos el buffer de los mensajes ya procesados
        for m in to_remove:
            Message.buffer.remove(m)

    def handle_message(self, msg):
        """
        Lógica para cada tipo de mensaje que este semáforo pueda recibir.
        - "request_green": VehicleAgent pide luz.
        - "turning_green": Otro semáforo avisa que se puso en verde.
        """
        if msg.perf == "request_green":
            logging.debug(
                f"[TrafficLightAgent {self.id}] Recibido request_green de Vehicle {msg.sender}"
            )
            can_go_green = self.can_turn_green()
            if can_go_green and not self.busy:
                # Iniciar ciclo en verde
                self.phase = "green"
                self.busy = True
                self.timer = self.green_t
                logging.debug(
                    f"[TrafficLightAgent {self.id}] -> GREEN (por request de Vehículo {msg.sender})"
                )

                # Avisar a otros semáforos que pasamos a verde
                broadcast = Message(
                    sender=self.id,
                    receiver=None,  # broadcast
                    perf="turning_green",
                    content=self.id,  # avisamos quién se pone verde
                )
                broadcast.send()

        elif msg.perf == "turning_green":
            # Otro semáforo se puso en verde
            if msg.content != self.id:
                # Si es distinto a mí y yo estaba en verde/amarillo, me pongo rojo
                if self.phase in ("green", "yellow"):
                    self.phase = "red"
                    self.busy = False
                    self.timer = 0
                    logging.debug(
                        f"[TrafficLightAgent {self.id}] -> RED (otro semáforo {msg.content} se puso en verde)"
                    )

        else:
            logging.debug(
                f"[TrafficLightAgent {self.id}] Mensaje desconocido: {msg.perf}"
            )

    def can_turn_green(self):
        """Chequear si hay algún otro semáforo en fase verde o amarillo."""
        for tl in self.model.lights:
            if tl.id != self.id and (tl.phase == "green" or tl.phase == "yellow"):
                return False
        return True

    def update_light(self):
        """Lógica de transición de semáforo green->yellow->red."""
        if not self.busy:
            # Estamos en rojo esperando solicitud
            return

        # Estamos en ciclo verde/amarillo
        self.timer -= 1
        if self.timer <= 0:
            if self.phase == "green":
                # Pasamos a amarillo
                self.phase = "yellow"
                self.timer = self.yellow_t
                logging.debug(f"[TrafficLightAgent {self.id}] -> YELLOW")

            elif self.phase == "yellow":
                # Terminamos ciclo y pasamos a rojo
                self.phase = "red"
                self.busy = False
                logging.debug(f"[TrafficLightAgent {self.id}] -> RED (fin ciclo)")


###############################################################################
#                           AGENTE VEHÍCULO                                   #
###############################################################################


def dynamic_deceleration(dist, max_dist=10, max_decel=25.0):
    """
    Devuelve una desaceleración (un valor positivo que indica cuánto se resta
    a la velocidad) basada en la cercanía.
      - dist > max_dist => 0 (sin frenada adicional)
      - dist = 0 => max_decel (frenada máxima)
      - dist en (0, max_dist] => escala lineal [max_decel..0]
    """
    if dist <= 0:
        return max_decel
    if dist >= max_dist:
        return 0
    factor = 1.0 - (dist / max_dist)
    return max_decel * factor


class VehicleAgent(ap.Agent):
    def setup(self):
        # 1. Construir lista de edges que no son intersecciones
        non_intersection_edges = []
        for s_node, conn_list in EDGES.items():
            for d_node, road_type in conn_list:
                if road_type != "interseccion":
                    # Guardamos: (origen, destino, tipo_de_calle)
                    non_intersection_edges.append((s_node, d_node, road_type))

        # 2. Escogemos uno de esos edges al azar
        start_s, start_d, start_type = random.choice(non_intersection_edges)

        # 3. Hallamos la posición de los nodos para interpolar
        p_s = graph.get_node(start_s).pos  # (x,0,z) del nodo origen
        p_d = graph.get_node(start_d).pos  # (x,0,z) del nodo destino

        # 4. Elegir un porcentaje 't' aleatorio en [0, 1]
        t = random.random()

        # Interpolación para ubicar el vehículo en un punto intermedio
        x = p_s[0] + (p_d[0] - p_s[0]) * t
        z = p_s[2] + (p_d[2] - p_s[2]) * t
        y = p_s[1]  # normalmente será 0, pero dejamos por consistencia

        self.position = (x, y, z)

        # 5. Guardamos la info para que el vehículo "encaje" con la lógica actual
        self.node = start_s  # Este "nodo" es el origen del edge
        self.current_road_type = start_type
        self.current_edge = (start_s, start_d)

        # El vehículo requiere saber cuál es su "start_pos" y "end_pos"
        self.start_pos = (p_s[0], y, p_s[2])
        self.end_pos = (p_d[0], y, p_d[2])

        # Distancia total del tramo
        dx = p_d[0] - p_s[0]
        dz = p_d[2] - p_s[2]
        self.dist = math.hypot(dx, dz)

        # El 'prog' indica el progreso en la calle (0 = inicio, 1 = final).
        # Si lo iniciamos en 't', significa que ya estamos en esa fracción del tramo.
        self.prog = t

        # Valores de velocidad y aceleración
        self.speed = 1
        self.max_speed = 5
        self.acc = 0.5
        self.finished = False

        logging.debug(
            f"[VehicleAgent #{self.id}] setup -> edge=({start_s}->{start_d}), t={t:.2f}"
        )

    def step(self):
        if self.finished:
            return

        # 1. Procesar mensajes (si hubiera)
        self.receive_messages()

        # 2. Cálculo de frenada total
        dec_total = 0.0

        # (a) Vehículo adelante
        dec_total += self.sense_vehicle_ahead()

        # (b) Semáforo rojo (si es calle tipo "luz")
        dec_total += self.handle_traffic_light_if_needed()

        # 3. Aplicar frenado o aceleración
        if dec_total > 0:
            new_speed = self.speed - dec_total
            self.speed = max(0, new_speed)
            logging.debug(
                f"[VehicleAgent #{self.id}] BRAKE -> speed={self.speed:.2f} (dec={dec_total:.2f})"
            )
        else:
            old_speed = self.speed
            self.speed = min(self.speed + self.acc, self.max_speed)
            if self.speed > old_speed:
                logging.debug(
                    f"[VehicleAgent #{self.id}] ACCEL -> speed={self.speed:.2f}"
                )

        # 4. Movernos
        self.move()

    def receive_messages(self):
        """Lee cualquier mensaje dirigido a este vehículo (actualmente no se usa)."""
        to_remove = []
        for msg in Message.buffer:
            if msg.receiver == self.id:
                logging.debug(f"[VehicleAgent #{self.id}] Mensaje recibido: {msg.perf}")
                to_remove.append(msg)
        for m in to_remove:
            Message.buffer.remove(m)

    def sense_vehicle_ahead(self):
        """
        Devuelve la desaceleración por tener un vehículo
        muy cerca (d < 10) en la misma dirección de avance.
        """
        if self.finished or self.prog >= 1:
            return 0.0

        dir_x = self.end_pos[0] - self.position[0]
        dir_z = self.end_pos[2] - self.position[2]
        dir_len = math.hypot(dir_x, dir_z)
        if dir_len < 1e-9:
            return 0.0
        dir_x /= dir_len
        dir_z /= dir_len

        min_dist = None
        for o in self.model.vehicles:
            if o.id == self.id or o.finished:
                continue
            ox = o.position[0] - self.position[0]
            oz = o.position[2] - self.position[2]
            dot = (dir_x * ox) + (dir_z * oz)
            if dot > 0:
                dist_veh = math.hypot(ox, oz)
                if dist_veh < 10:
                    if (min_dist is None) or (dist_veh < min_dist):
                        min_dist = dist_veh

        if min_dist is not None:
            dec = dynamic_deceleration(min_dist, max_dist=10, max_decel=20.0)
            return dec
        return 0.0

    def handle_traffic_light_if_needed(self):
        """
        Envía 'request_green' al semáforo si es una calle tipo 'luz'.
        Si está en rojo, frenamos en función de la cercanía al final del tramo.
        """
        if self.current_road_type != "luz" or self.current_edge is None:
            return 0.0

        # Buscamos el semáforo correspondiente
        for tl in self.model.lights:
            if tl.edge == self.current_edge:
                # Enviar solicitud de verde
                msg = Message(
                    sender=self.id,
                    receiver=tl.id,
                    perf="request_green",
                    content={"edge": self.current_edge},
                )
                msg.send()
                # Frenar si está rojo
                if tl.phase == "red":
                    dist_to_end = self.dist * (1 - self.prog)
                    dec = dynamic_deceleration(
                        dist=dist_to_end, max_dist=self.dist, max_decel=20.0
                    )
                    logging.debug(
                        f"[VehicleAgent #{self.id}] Semáforo {tl.id} ROJO. dist_end={dist_to_end:.1f}, dec={dec:.2f}"
                    )
                    return dec
                return 0.0
        return 0.0

    def move(self):
        if self.finished:
            return

        distance_step = (
            self.speed
        )  # lo que avanzaríamos este step (en unidades de espacio)
        still_to_move = (
            distance_step  # nos queda esta distancia por avanzar en esta “ronda”
        )

        # Podríamos usar un while por si en el mismo step recorremos más de un tramo.
        # (Imagina tramos muy cortos y velocidad alta)
        while still_to_move > 1e-9 and not self.finished:

            # ¿Cuánta distancia falta para terminar el tramo actual?
            dist_remaining = self.dist * (1 - self.prog)

            if still_to_move >= dist_remaining:
                # Llegamos (o sobrepasamos) el final de la calle en este step
                self.prog = 1
                self.position = (self.end_pos[0], self.start_pos[1], self.end_pos[2])
                self.node = self.find_node(self.position)

                # Restamos lo que hemos avanzado
                still_to_move -= dist_remaining

                # Elegir el siguiente tramo
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

                # Ahora que hay una nueva calle, reiniciamos prog,
                # pero OJO: no empezamos en 0, sino que enseguida avanzamos
                self.prog = 0

            else:
                # No llegamos al final de la calle actual. Avanzamos "still_to_move" sobre ella
                # Convertimos esa distancia lineal "still_to_move" en fracción (delta_prog)
                delta_prog = still_to_move / (self.dist + 1e-9)
                self.prog += delta_prog
                # Nos quedamos en este tramo, no pasamos a la siguiente
                still_to_move = 0

            # Actualizar posición si prog < 1
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
        """Encuentra un nodo cuyo (x,z) sea muy parecido a p."""
        for name, nd in graph.nodes.items():
            if abs(nd.pos[0] - p[0]) < 1e-3 and abs(nd.pos[2] - p[2]) < 1e-3:
                return name
        return None


###############################################################################
#                               MODELO PRINCIPAL                              #
###############################################################################


class TwoTypeStreetsModel(ap.Model):
    def setup(self):
        logging.info("TwoTypeStreetsModel.setup() called (mensajes).")
        # Crear semáforos
        self.lights = ap.AgentList(self, len(LIGHTS_INFO), TrafficLightAgent)
        for i, tl in enumerate(self.lights):
            tl.edge = LIGHTS_INFO[i]["edge"]
            tl.pos = LIGHTS_INFO[i]["pos"]
            logging.debug(
                f"Created TrafficLightAgent {tl.id} for edge={tl.edge} at {tl.pos}"
            )
        # Crear vehículos
        self.vehicles = ap.AgentList(self, NUM_VEHICLES, VehicleAgent)

    def step(self):
        logging.debug("===== TwoTypeStreetsModel step() =====")
        self.lights.step()
        self.vehicles.step()


###############################################################################
#                           VISUALIZACIÓN OPENGL                              #
###############################################################################


def draw_axes():
    gl.glBegin(gl.GL_LINES)
    # Eje X (rojo)
    gl.glColor3f(1, 0, 0)
    gl.glVertex3f(-50, 0, 0)
    gl.glVertex3f(50, 0, 0)
    # Eje Y (verde)
    gl.glColor3f(0, 1, 0)
    gl.glVertex3f(0, -50, 0)
    gl.glVertex3f(0, 50, 0)
    # Eje Z (azul)
    gl.glColor3f(0, 0, 1)
    gl.glVertex3f(0, 0, -50)
    gl.glVertex3f(0, 0, 50)
    gl.glEnd()


def draw_lines():
    # yellow line from (-2, -85) to (-1, -30) and another from (1, -85) to (2, -30)
    # the same for the other 3 sides
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
    # we draw 4 lines for each rectangle
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
    """
    Dibuja cada edge según su tipo:
    - 'normal': dibuja un rectángulo de ancho 8 (4u a cada lado).
    - 'luz' o 'interseccion': dibuja simplemente una línea.
    """
    for start_node, connections in EDGES.items():
        p1 = graph.get_node(start_node).pos  # (x, 0, z)
        x1, _, z1 = p1

        for dest_node, label in connections:
            p2 = graph.get_node(dest_node).pos
            x2, _, z2 = p2

            # Vector dirección (en xz)
            dx = x2 - x1
            dz = z2 - z1
            length = math.hypot(dx, dz)
            if length < 1e-9:
                continue  # evita división por cero

            # Vector dirección normalizado
            ndx = dx / length
            ndz = dz / length

            # Vector perpendicular (en el plano XZ) => (-dz, dx)
            px = -ndz
            pz = ndx

            # Offset de 10 unidades a cada lado (total = 20 de ancho)
            offset = 10.0
            ox = px * offset
            oz = pz * offset

            if label == "normal" or label == "luz":
                # DIBUJAR RECTÁNGULO
                p1_left = (x1 - ox, 0, z1 - oz)
                p1_right = (x1 + ox, 0, z1 + oz)
                p2_left = (x2 - ox, 0, z2 - oz)
                p2_right = (x2 + ox, 0, z2 + oz)

                gl.glColor3f(0.5, 0.5, 0.5)  # gris
                gl.glBegin(gl.GL_QUADS)
                gl.glVertex3f(*p1_left)
                gl.glVertex3f(*p1_right)
                gl.glVertex3f(*p2_right)
                gl.glVertex3f(*p2_left)
                gl.glEnd()

            else:
                # DIBUJAR LÍNEA con efecto dotted color rojo
                gl.glColor3f(1, 0, 0)  # rojo
                gl.glLineStipple(1, 0xAAAA)
                gl.glEnable(gl.GL_LINE_STIPPLE)
                gl.glBegin(gl.GL_LINES)
                gl.glVertex3f(x1, 1, z1)
                gl.glVertex3f(x2, 1, z2)
                gl.glEnd()
                gl.glDisable(gl.GL_LINE_STIPPLE)


def draw_vehicle(pos, finished):
    if finished:
        return
    gl.glPushMatrix()
    gl.glTranslatef(pos[0], pos[1], pos[2])
    gl.glColor3f(0, 1, 1)  # cian para los vehículos
    glut.glutSolidSphere(4, 16, 16)
    gl.glPopMatrix()


def draw_light(tl):
    """
    Dibuja 3 esferas (focos) en la posición del semáforo,
    todas con el color actual (verde/amarillo/rojo).
    """
    x, y, z = tl.pos
    gl.glPushMatrix()
    gl.glTranslatef(x, y, z)
    if tl.phase == "green":
        gl.glColor3f(0, 1, 0)
    elif tl.phase == "yellow":
        gl.glColor3f(1, 1, 0)
    else:  # red
        gl.glColor3f(1, 0, 0)

    # Simplemente dibujamos 3 focos con un pequeño offset en Y
    for offset in [5, 0, -5]:
        gl.glPushMatrix()
        gl.glTranslatef(0, offset, 0)
        glut.glutSolidSphere(1, 16, 16)
        gl.glPopMatrix()
    gl.glPopMatrix()


###############################################################################
#           NUEVA FUNCIÓN: DIBUJAR ÁREAS (CUADRADO Y RECTÁNGULOS)            #
###############################################################################
def draw_areas():
    # 1. Cuadrado gris claro de (-15, -15) a (15, 15)
    gl.glColor3f(0.8, 0.8, 0.8)  # gris claro
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex3f(X[2], 0, X[2])  # (-15, -15)
    gl.glVertex3f(X[5], 0, X[2])  # ( 15, -15)
    gl.glVertex3f(X[5], 0, X[5])  # ( 15,  15)
    gl.glVertex3f(X[2], 0, X[5])  # (-15,  15)
    gl.glEnd()

    # 2. Rectángulos verdes
    gl.glColor3f(0.0, 1.0, 0.0)  # verde

    # (a) de (-85, -85) a (-15, -15)
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex3f(-85, 0, -85)
    gl.glVertex3f(X[2], 0, -85)  # -15
    gl.glVertex3f(X[2], 0, X[2])  # -15
    gl.glVertex3f(-85, 0, X[2])
    gl.glEnd()

    # (b) de (-85, 15) a (-15, 85)
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex3f(-85, 0, X[5])  # 15
    gl.glVertex3f(X[2], 0, X[5])  # -15, 15
    gl.glVertex3f(X[2], 0, 85)
    gl.glVertex3f(-85, 0, 85)
    gl.glEnd()

    # (c) de (15, -85) a (85, -15)
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex3f(X[5], 0, -85)  # 15
    gl.glVertex3f(85, 0, -85)
    gl.glVertex3f(85, 0, X[2])  # -15
    gl.glVertex3f(X[5], 0, X[2])  # 15, -15
    gl.glEnd()

    # (d) de (15, 15) a (85, 85)
    gl.glBegin(gl.GL_QUADS)
    gl.glVertex3f(X[5], 0, X[5])  # 15, 15
    gl.glVertex3f(85, 0, X[5])  # 85, 15
    gl.glVertex3f(85, 0, 85)
    gl.glVertex3f(X[5], 0, 85)
    gl.glEnd()


def draw_crosswalk(p1, p2, width=10, num_stripes=8):
    """
    Dibuja un paso de cebra entre los puntos p1 y p2 (cada uno (x,y,z)),
    con 'num_stripes' franjas y 'width' de ancho total.
    Alterna franjas blancas opacas y blancas translúcidas.
    """

    # Activamos blending para la transparencia
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    # Extraemos coordenadas
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    # Vector dirección (solo en XZ)
    dx = x2 - x1
    dz = z2 - z1
    length = math.hypot(dx, dz)
    if length < 1e-9:
        gl.glDisable(gl.GL_BLEND)
        return  # Evitamos división por cero si el tramo es muy pequeño

    # Normalizamos el vector dirección
    ndx = dx / length
    ndz = dz / length

    # Vector perpendicular en XZ (para el ancho)
    # Si la dirección es (ndx, ndz), la perpendicular puede ser (-ndz, ndx)
    # o (ndz, -ndx). Usaremos (ndz, -ndx) arbitrariamente.
    px = ndz
    pz = -ndx

    half_w = width / 2.0

    # Coordenadas "laterales" iniciales (p1_left, p1_right)
    # p1_left = p1 - half_w*(px, pz)
    # p1_right= p1 + half_w*(px, pz)
    p1_left = (x1 - half_w * px, y1, z1 - half_w * pz)
    p1_right = (x1 + half_w * px, y1, z1 + half_w * pz)

    # Análogamente para el punto final (p2_left, p2_right)
    p2_left = (x2 - half_w * px, y2, z2 - half_w * pz)
    p2_right = (x2 + half_w * px, y2, z2 + half_w * pz)

    # La longitud se parte en 'num_stripes' franjas
    stripe_len = length / num_stripes

    # Recorremos cada franja
    # Usamos un bucle para moverse de 0 a length en incrementos de stripe_len
    traveled = 0.0
    for i in range(num_stripes):
        # Calculamos el inicio y fin de esta franja en [0..length]
        start_dist = traveled
        end_dist = traveled + stripe_len

        # Interpolamos la fracción [start_dist/length .. end_dist/length]
        s1 = start_dist / length
        s2 = end_dist / length

        # Obtenemos los 4 vértices de la franja interpolando linealmente
        # entre p1_left->p2_left y p1_right->p2_right
        f1_left_x = p1_left[0] + (p2_left[0] - p1_left[0]) * s1
        f1_left_y = p1_left[1] + (p2_left[1] - p1_left[1]) * s1
        f1_left_z = p1_left[2] + (p2_left[2] - p1_left[2]) * s1

        f1_right_x = p1_right[0] + (p2_right[0] - p1_right[0]) * s1
        f1_right_y = p1_right[1] + (p2_right[1] - p1_right[1]) * s1
        f1_right_z = p1_right[2] + (p2_right[2] - p1_right[2]) * s1

        f2_left_x = p1_left[0] + (p2_left[0] - p1_left[0]) * s2
        f2_left_y = p1_left[1] + (p2_left[1] - p1_left[1]) * s2
        f2_left_z = p1_left[2] + (p2_left[2] - p1_left[2]) * s2

        f2_right_x = p1_right[0] + (p2_right[0] - p1_right[0]) * s2
        f2_right_y = p1_right[1] + (p2_right[1] - p1_right[1]) * s2
        f2_right_z = p1_right[2] + (p2_right[2] - p1_right[2]) * s2

        # Asignamos el color de la franja
        # Par: blanco opaco, Impar: blanco semi-transparente
        if i % 2 == 0:
            gl.glColor4f(1.0, 1.0, 1.0, 1.0)  # Blanco opaco
        else:
            gl.glColor4f(1.0, 1.0, 1.0, 0.0)  # Blanco translúcido

        # Dibujamos el quad
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex3f(f1_left_x, 0.1, f1_left_z)
        gl.glVertex3f(f1_right_x, 0.1, f1_right_z)
        gl.glVertex3f(f2_right_x, 0.1, f2_right_z)
        gl.glVertex3f(f2_left_x, 0.1, f2_left_z)
        gl.glEnd()

        traveled += stripe_len

    gl.glDisable(gl.GL_BLEND)


def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    # Cámara
    glu.gluLookAt(0, 150, 150, 0, 0, 0, 0, 1, 0)

    draw_areas()
    draw_axes()
    draw_lines()
    draw_roads()

    # ---- AQUI AGREGAMOS EL PASO DE CEBRA ----
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
        draw_vehicle(v.position, v.finished)

    # Semáforos
    for tl in model.lights:
        draw_light(tl)

    glut.glutSwapBuffers()


def update(_):
    model.step()
    glut.glutPostRedisplay()
    # Llamar a 'update' cada 100 ms
    glut.glutTimerFunc(100, update, 0)


def main():
    logging.info(
        "main() -> Iniciando simulación con mensajes (carro-semaforo y semaforo-semaforo)."
    )
    glut.glutInit()
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB | glut.GLUT_DEPTH)
    glut.glutInitWindowSize(800, 600)
    glut.glutCreateWindow(b"Calles normales e iluminadas - Mensajes")

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glClearColor(0, 0, 0, 1)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(60, 800 / 600, 1, 1000)

    glut.glutDisplayFunc(display)
    glut.glutTimerFunc(100, update, 0)

    # Inicializamos el modelo global
    global model
    model = TwoTypeStreetsModel({})
    model.setup()

    logging.info("main() -> Entrando a glutMainLoop.")
    glut.glutMainLoop()


if __name__ == "__main__":
    main()
