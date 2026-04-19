import math


class HazardAnalyzer:
    def __init__(self):
        self.previous_tracks = []
        self.next_track_id = 0

        self.match_distance_threshold = 80

        self.close_area_threshold = 0.12
        self.medium_area_threshold = 0.04

        self.approaching_threshold = 0.15
        self.moving_away_threshold = -0.15

        self.vehicle_labels = {"car", "truck", "bus", "motorcycle", "bicycle"}
        self.obstacle_labels = {"cone", "barrier", "trash bag", "branch"}
        self.person_labels = {"person"}
        self.ground_hazard_labels = {"pothole"}

    def analyze(self, detections, frame_width, frame_height):
        detections = self._assign_positions(detections, frame_width)
        detections = self._assign_proximity(detections, frame_width, frame_height)
        detections = self._match_with_previous(detections)
        detections = self._assign_motion_trend(detections)
        detections = self._assign_hazard_level(detections)
        self._update_tracks(detections)
        return detections

    def _assign_positions(self, detections, frame_width):
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            center_x = (x1 + x2) / 2

            if center_x < frame_width / 3:
                det["position"] = "left"
            elif center_x < 2 * frame_width / 3:
                det["position"] = "center"
            else:
                det["position"] = "right"

        return detections

    def _assign_proximity(self, detections, frame_width, frame_height):
        frame_area = frame_width * frame_height

        for det in detections:
            x1, y1, x2, y2 = det["box"]

            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)
            area = box_w * box_h

            relative_area = area / frame_area if frame_area > 0 else 0

            det["box_area"] = area
            det["relative_area"] = relative_area

            if relative_area > self.close_area_threshold:
                det["proximity"] = "close"
            elif relative_area > self.medium_area_threshold:
                det["proximity"] = "medium"
            else:
                det["proximity"] = "far"

        return detections

    def _box_center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _match_with_previous(self, detections):
        used_previous_ids = set()

        for det in detections:
            det["track_id"] = None
            det["previous_area"] = None

            current_label = det["label"].lower()
            cx, cy = self._box_center(det["box"])

            best_match = None
            best_distance = float("inf")

            for prev in self.previous_tracks:
                if prev["label"] != current_label:
                    continue

                if prev["track_id"] in used_previous_ids:
                    continue

                px, py = prev["center"]
                distance = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

                if distance < self.match_distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match = prev

            if best_match is not None:
                det["track_id"] = best_match["track_id"]
                det["previous_area"] = best_match["box_area"]
                used_previous_ids.add(best_match["track_id"])
            else:
                det["track_id"] = self.next_track_id
                self.next_track_id += 1

        return detections

    def _assign_motion_trend(self, detections):
        for det in detections:
            prev_area = det.get("previous_area")
            curr_area = det.get("box_area", 0)

            if prev_area is None or prev_area == 0:
                det["motion"] = "unknown"
                continue

            growth = (curr_area - prev_area) / prev_area

            if growth > self.approaching_threshold:
                det["motion"] = "approaching"
            elif growth < self.moving_away_threshold:
                det["motion"] = "moving_away"
            else:
                det["motion"] = "stable"

        return detections

    def _assign_hazard_level(self, detections):
        for det in detections:
            label = det["label"].lower()
            position = det["position"]
            proximity = det["proximity"]
            motion = det["motion"]

            severity = "ignore"

            if label in self.vehicle_labels:
                if proximity == "close" and position == "center" and motion in {"approaching", "stable"}:
                    severity = "hazard"
                elif (proximity == "medium" and position == "center") or (proximity == "close" and position in {"left", "right"}):
                    severity = "caution"

            elif label in self.obstacle_labels:
                if proximity == "close" and position == "center":
                    severity = "hazard"
                elif proximity in {"medium", "close"}:
                    severity = "caution"

            elif label in self.person_labels:
                if proximity == "close" and position == "center" and motion == "approaching":
                    severity = "hazard"
                elif proximity in {"medium", "close"} and position == "center":
                    severity = "caution"

            elif label in self.ground_hazard_labels:
                if proximity == "close" and position == "center":
                    severity = "hazard"
                elif proximity in {"medium", "close"}:
                    severity = "caution"

            det["severity"] = severity
            det["is_hazard"] = severity in {"hazard", "caution"}

        return detections

    def _update_tracks(self, detections):
        new_tracks = []

        for det in detections:
            cx, cy = self._box_center(det["box"])

            new_tracks.append({
                "track_id": det["track_id"],
                "label": det["label"].lower(),
                "center": (cx, cy),
                "box_area": det["box_area"]
            })

        self.previous_tracks = new_tracks