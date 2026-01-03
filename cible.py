import pygame
import cv2
import numpy as np
import threading
import queue

# ===================== CONSTANTES =====================
CM_TO_PX = 37.7953
TARGET_WIDTH_CM = 14
TARGET_HEIGHT_CM = 14.5
TARGET_WIDTH_PX = int(TARGET_WIDTH_CM * CM_TO_PX)
TARGET_HEIGHT_PX = int(TARGET_HEIGHT_CM * CM_TO_PX)

RING_STEP_CM = 0.5
RING_STEP_PX = RING_STEP_CM * CM_TO_PX
NUM_OFFSET_PX = int(0.25 * CM_TO_PX)

MIN_CONTOUR_AREA = 25
MAX_CONTOUR_AREA = 500


# ===================== INIT =====================
pygame.init()
cap = cv2.VideoCapture(0)
cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
screen = pygame.display.set_mode((cam_w + TARGET_WIDTH_PX + 40, cam_h))
pygame.display.set_caption("Webcam + Calibration + Cible virtuelle")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# ===================== CALIBRATION =====================
points = [[100, 100], [300, 100], [300, 300], [100, 300]]
selected_point = None
rayon_souris = 15
center_point = [200, 200]
selected_center = False

impact_positions = []

# ===================== MODE =====================
mode_calibration = True  # début en calibration

# ===================== THREAD TERMINAL =====================
command_queue = queue.Queue()
def read_terminal():
    while True:
        cmd = input("Choisir : 0=changer mode calibration, 1=réinitialiser impacts\n> ")
        command_queue.put(cmd)

threading.Thread(target=read_terminal, daemon=True).start()

# ===================== INIT IMAGE PRECEDENTE =====================
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# ===================== BOUCLE PRINCIPALE =====================
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------- EVENTS PYGAME --------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            for i, (px, py) in enumerate(points):
                if abs(mx - px) < rayon_souris and abs(my - py) < rayon_souris:
                    selected_point = i
            if abs(mx - center_point[0]) < rayon_souris and abs(my - center_point[1]) < rayon_souris:
                selected_center = True
        elif event.type == pygame.MOUSEBUTTONUP:
            selected_point = None
            selected_center = False
        elif event.type == pygame.MOUSEMOTION:
            mx, my = pygame.mouse.get_pos()
            if selected_point is not None:
                points[selected_point] = [mx, my]
            if selected_center:
                center_point = [mx, my]

    # -------- COMMANDES TERMINAL --------
    while not command_queue.empty():
        cmd = command_queue.get()
        if cmd == "0":
            mode_calibration = not mode_calibration
            print(f"Mode calibration : {mode_calibration}")
        elif cmd == "1":
            impact_positions = []
            print("Impacts rouges réinitialisés")


    # -------- DÉTECTION IMPACTS (hors calibration) --------
    if not mode_calibration:
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [np.array(points)], 255)
        diff = cv2.absdiff(gray, prev_gray)
        diff = cv2.bitwise_and(diff, mask)

        _, diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                ratio = w / h if h != 0 else 0
                if 0.5 <= ratio <= 2:
                    impact_positions.append((x + w // 2, y + h // 2))

        prev_gray = gray.copy()

    # -------- DESSIN SUR FRAME --------
    cv2.polylines(frame, [np.array(points)], True, (0, 255, 0), 2)
    for px, py in points:
        cv2.circle(frame, (px, py), 5, (255, 0, 0), -1)
    for ix, iy in impact_positions:
        cv2.circle(frame, (ix, iy), 6, (0, 0, 255), 2)
    cv2.circle(frame, tuple(center_point), 6, (0, 255, 255), -1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cam_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    screen.blit(cam_surface, (0, 0))

    # -------- CIBLE VIRTUELLE --------
    target_x = cam_w + 20
    target_y = (cam_h - TARGET_HEIGHT_PX) // 2
    pygame.draw.rect(screen, (150, 150, 150), (target_x, target_y, TARGET_WIDTH_PX, TARGET_HEIGHT_PX))
    center_x = target_x + TARGET_WIDTH_PX // 2
    center_y = target_y + TARGET_HEIGHT_PX // 2
    offsets = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    for i in range(1, 11):
        radius = int((11 - i) * RING_STEP_PX)
        if 5 <= i <= 7:
            fill_color = (0, 0, 0)
            border_color = (255, 255, 255)
            num_color = (255, 255, 255)
        else:
            fill_color = (255, 255, 255)
            border_color = (0, 0, 0)
            num_color = (0, 0, 0)

        pygame.draw.circle(screen, fill_color, (center_x, center_y), radius)
        pygame.draw.circle(screen, border_color, (center_x, center_y), radius, 2)

        if i <= 9:
            for ox, oy in offsets:
                text = font.render(str(i), True, num_color)
                text_rect = text.get_rect(center=(
                    int(center_x + ox * (radius - NUM_OFFSET_PX)),
                    int(center_y + oy * (radius - NUM_OFFSET_PX))
                ))
                screen.blit(text, text_rect)

    # -------- IMPACTS SUR CIBLE VIRTUELLE --------
    cam_rect_width = max(p[0] for p in points) - min(p[0] for p in points)
    cam_rect_height = max(p[1] for p in points) - min(p[1] for p in points)
    scale_x = TARGET_WIDTH_PX / cam_rect_width
    scale_y = TARGET_HEIGHT_PX / cam_rect_height

    for ix, iy in impact_positions:
        rel_x = ix - center_point[0]
        rel_y = iy - center_point[1]
        virt_x = int(center_x + rel_x * scale_x)
        virt_y = int(center_y + rel_y * scale_y)
        pygame.draw.circle(screen, (255, 0, 0), (virt_x, virt_y), 6, 2)

    pygame.display.flip()
    clock.tick(30)

cap.release()
pygame.quit()
