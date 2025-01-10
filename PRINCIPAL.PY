import cv2
import pygame
import numpy as np
import mediapipe as mp
from djitellopy import tello
from ultralytics import YOLO
from keyboard_control import getKeyboardInput

# Inicializando o Pygame e a tela
pygame.init()
screen = pygame.display.set_mode((640, 480))

# Inicializando o Haar Cascade para rosto (somente rosto humano)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializando MediaPipe para pose humana
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inicializando MediaPipe para mãos (detecção de mãos humanas)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Carrega o modelo YOLOv8 pré-treinado
model = YOLO('yolov8n.pt')  # Use o modelo YOLOv8 (ou outro modelo desejado)

# Inicializando o drone Tello
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

# Inicializando a captura de vídeo do drone
frame_read = me.get_frame_read()  # Obtém o objeto para ler os quadros do drone

# Controle pelo teclado
def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    for event in pygame.event.get(): pass  # Limpar eventos anteriores
    keyInput = pygame.key.get_pressed()

    if keyInput[pygame.K_LEFT]: lr = -speed
    elif keyInput[pygame.K_RIGHT]: lr = speed

    if keyInput[pygame.K_UP]: fb = speed
    elif keyInput[pygame.K_DOWN]: fb = -speed

    if keyInput[pygame.K_w]: ud = speed
    elif keyInput[pygame.K_s]: ud = -speed

    if keyInput[pygame.K_a]: yv = speed
    elif keyInput[pygame.K_d]: yv = -speed

    if keyInput[pygame.K_q]: me.land()
    if keyInput[pygame.K_e]: me.takeoff()

    return [lr, fb, ud, yv]

# Definindo a taxa de quadros para limitar a velocidade da exibição
fps_limit = 30
clock = pygame.time.Clock()

# Função para detectar e exibir as pessoas
def detect_and_display():
    frame = frame_read.frame  # Captura o quadro do drone
    if frame is None:
        print("Erro ao capturar a imagem.")
        return

    # Reduzindo a resolução do vídeo para melhorar o desempenho
    frame_resized = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Detectando poses humanas com MediaPipe
    results = pose.process(frame_rgb)
    bounding_boxes = []

    # Processando resultados do MediaPipe Pose para identificar limites de pessoas
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        x_min = min([int(landmark.x * frame_resized.shape[1]) for landmark in landmarks])
        y_min = min([int(landmark.y * frame_resized.shape[0]) for landmark in landmarks])
        x_max = max([int(landmark.x * frame_resized.shape[1]) for landmark in landmarks])
        y_max = max([int(landmark.y * frame_resized.shape[0]) for landmark in landmarks])
        bounding_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    # Adiciona detecção do Haar Cascade para faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detectando mãos com MediaPipe Hands
    hand_results = hands.process(frame_rgb)
    hand_bounding_boxes = []

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            x_min = min([int(hand_landmark.x * frame_resized.shape[1]) for hand_landmark in hand_landmarks.landmark])
            y_min = min([int(hand_landmark.y * frame_resized.shape[0]) for hand_landmark in hand_landmarks.landmark])
            x_max = max([int(hand_landmark.x * frame_resized.shape[1]) for hand_landmark in hand_landmarks.landmark])
            y_max = max([int(hand_landmark.y * frame_resized.shape[0]) for hand_landmark in hand_landmarks.landmark])
            hand_bounding_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    # Detectando pessoas com YOLO
    results_yolo = model(frame_resized)
    yolo_bounding_boxes = []

    # Filtrando as detecções para "pessoas" (classe 0)
    for result in results_yolo:
        for box in result.boxes:
            if box.cls == 0:  # Classe 0 é para "pessoa"
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Pega as coordenadas da caixa
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Converte para inteiros
                yolo_bounding_boxes.append((x1, y1, x2 - x1, y2 - y1))

    # Mesclando as detecções em uma única caixa por pessoa detectada
    unified_boxes = []
    for box in bounding_boxes + hand_bounding_boxes + yolo_bounding_boxes:
        x, y, w, h = box
        found = False
        for ubox in unified_boxes:
            ux, uy, uw, uh = ubox
            # Melhorar a lógica de sobreposição para garantir que partes diferentes da pessoa sejam unificadas
            if x < ux + uw and x + w > ux and y < uy + uh and y + h > uy:  # Se há sobreposição
                ubox = (min(ux, x), min(uy, y), max(ux + uw, x + w) - min(ux, x), max(uy + uh, y + h) - min(uy, y))
                found = True
                break
        if not found:
            unified_boxes.append(box)

    # Contador de pessoas (número de caixas unificadas)
    person_count = len(unified_boxes)

    # Desenhando os retângulos unificados em torno das pessoas detectadas e adicionando o rótulo
    for idx, (x, y, w, h) in enumerate(unified_boxes):
        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Pessoa {idx + 1}"
        cv2.putText(frame_resized, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Exibindo o contador de pessoas no canto superior esquerdo
    cv2.putText(frame_resized, f"Pessoas detectadas: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Corrige a ordem das cores para o Pygame e converte para a superfície
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))

    # Exibindo o frame no Pygame
    screen.blit(frame_surface, (0, 0))
    pygame.display.flip()

    # Limita a taxa de quadros para 30fps
    clock.tick(fps_limit)

# Loop principal
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    detect_and_display()
