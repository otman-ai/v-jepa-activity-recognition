"""Reusable overlay matching the dashboard design in the notebook."""

import cv2
import numpy as np


def draw_action_overlay(frame, action, confidence, frame_index, total_frames, fps):
    height, width = frame.shape[:2]
    scale = max(width / 1000, 1.0)
    margin = int(width * 0.035)
    white, muted, teal = (245, 248, 255), (175, 190, 210), (190, 210, 45)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def panel(x1, y1, x2, y2, alpha=0.78):
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return
        roi = frame[y1:y2, x1:x2]
        cv2.addWeighted(np.full_like(roi, (17, 24, 39)), alpha, roi, 1-alpha, 0, roi)
        cv2.rectangle(frame, (x1, y1), (x2-1, y2-1), teal, 2, cv2.LINE_AA)

    def text(value, x, y, size, color, thickness=2):
        cv2.putText(frame, str(value), (x, y), font, size*scale, color,
                    max(2, int(thickness*scale)), cv2.LINE_AA)

    card_h = int(height * 0.22)
    panel(margin, margin, width-margin, margin+card_h)
    left = margin + int(width*0.025)
    text("DETECTED ACTION", left, margin+int(card_h*.25), .65, teal, 2.5)
    label = str(action).replace("_", " ").upper()
    if len(label) > 25:
        label = label[:22] + "..."
    text(label, left, margin+int(card_h*.58), 1.35, white, 3.5)
    confidence = float(np.clip(confidence, 0, 100))
    text(f"{confidence:.1f}% CONFIDENCE", left, margin+int(card_h*.80), .68, teal, 2.5)
    bar_x, bar_y = left, margin+int(card_h*.88)
    bar_w, bar_h = int(width*.45), max(8, int(height*.012))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (65,75,90), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+int(bar_w*confidence/100), bar_y+bar_h), teal, -1)
    right = width-margin-int(width*.22)
    text("VIDEO ANALYSIS", right, margin+int(card_h*.40), .55, muted, 2)
    text(f"{frame_index+1}/{max(total_frames,1)}", right, margin+int(card_h*.59), .75, white, 2.5)
    text(f"{fps:.1f} FPS", right, margin+int(card_h*.77), .55, muted, 2)

    header_h = int(height*.115)
    header_y = height-header_h-margin
    panel(margin, header_y, width-margin, height-margin)
    accent_w = max(5, int(width*.006))
    cv2.rectangle(frame, (margin, header_y), (margin+accent_w, height-margin), teal, -1)
    text("AI ACTION RECOGNITION", margin+int(width*.025), header_y+int(header_h*.42), .85, teal, 2.5)
    text("V-JEPA2  |  POOLED MLP CLASSIFIER", margin+int(width*.025), header_y+int(header_h*.76), .58, white, 1.5)
    status_x = width-margin-int(width*.22)
    cv2.circle(frame, (status_x, header_y+header_h//2), max(6, int(8*scale)), teal, -1)
    text("ACTIVE", status_x+int(18*scale), header_y+header_h//2+int(6*scale), .55, white)
    progress = min(1.0, (frame_index+1)/max(total_frames,1))
    cv2.rectangle(frame, (0,height-max(6,int(height*.006))), (int(width*progress),height), teal, -1)
    return frame
