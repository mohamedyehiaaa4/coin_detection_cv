import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Coin Extractor", layout="wide")


def preprocess_image(uploaded_bytes, max_width=900):
    """Decode uploaded image bytes and create grayscale/blur versions."""
    image_array = np.frombuffer(uploaded_bytes, dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image.")

    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    return img, gray, blurred


def get_coin_size(radius, min_detected_radius, max_detected_radius):
    """Classify coin size relative to other coins in the same image."""
    if max_detected_radius <= min_detected_radius:
        return "Medium"
    normalized = (radius - min_detected_radius) / (
        max_detected_radius - min_detected_radius
    )
    if normalized < 1 / 3:
        return "Small"
    if normalized < 2 / 3:
        return "Medium"
    return "Large"


COIN_COLOR_PALETTE = {
    "Silver": (192, 192, 192),
    "Gold": (212, 175, 55),
    "Copper": (184, 115, 51),
    "Bronze": (205, 127, 50),
    "Dark": (75, 75, 75),
}


def rgb_to_bgr(rgb):
    """Convert an RGB color tuple to OpenCV's BGR order."""
    red, green, blue = rgb
    return blue, green, red


def classify_coin_color(image, x, y, radius, palette=COIN_COLOR_PALETTE):
    """Classify the visible coin color against a fixed RGB palette.

    The classifier samples the inner coin area to avoid noisy outer rims and
    compares the median sampled color to each palette color in LAB space, which
    is more stable than raw RGB distance under mild lighting changes.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    sample_radius = max(1, int(radius * 0.75))
    cv2.circle(mask, (int(x), int(y)), sample_radius, 255, -1)

    sampled_pixels = image[mask == 255]
    if sampled_pixels.size == 0:
        return "Unknown", "#000000"

    median_bgr = np.median(sampled_pixels, axis=0).astype(np.uint8)
    median_rgb = median_bgr[::-1]
    median_lab = cv2.cvtColor(np.uint8([[median_bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(
        float
    )

    best_name = None
    best_distance = float("inf")
    for name, rgb in palette.items():
        palette_bgr = np.uint8([[rgb_to_bgr(rgb)]])
        palette_lab = cv2.cvtColor(palette_bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        distance = float(np.linalg.norm(median_lab - palette_lab))
        if distance < best_distance:
            best_name = name
            best_distance = distance

    color_hex = "#{:02X}{:02X}{:02X}".format(*median_rgb)
    return best_name, color_hex


def detect_coins(
    preprocessed_image, hough_param1=50, hough_param2=34, min_radius=10, max_radius=100
):
    """Detect circles using multi-pass Hough + conservative fallback merge."""

    def edge_support_score(edges, x, y, r):
        rim_mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.circle(rim_mask, (x, y), r, 255, 2)
        rim_pixels = np.count_nonzero(rim_mask)
        if rim_pixels == 0:
            return 0.0
        edge_pixels = np.count_nonzero(cv2.bitwise_and(edges, edges, mask=rim_mask))
        return edge_pixels / rim_pixels

    def angular_edge_coverage(edges, x, y, r, samples=72):
        """Measure how continuously edges exist around the full circumference."""
        if r <= 0:
            return 0.0

        h, w = edges.shape[:2]
        hits = 0
        search_band = max(2, int(0.08 * r))

        for i in range(samples):
            theta = 2.0 * np.pi * i / samples
            found = False
            for dr in range(-search_band, search_band + 1):
                rr = r + dr
                if rr <= 0:
                    continue
                px = int(round(x + rr * np.cos(theta)))
                py = int(round(y + rr * np.sin(theta)))
                if px < 0 or py < 0 or px >= w or py >= h:
                    continue
                if edges[py, px] > 0:
                    found = True
                    break
            if found:
                hits += 1
        return hits / float(samples)

    def overlap_ratio(c1, c2):
        x1, y1, r1 = c1
        x2, y2, r2 = c2
        d = float(np.hypot(x1 - x2, y1 - y2))
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            return 1.0
        return max(0.0, (r1 + r2 - d) / (2.0 * min(r1, r2)))

    def suppress_duplicates(candidates):
        if not candidates:
            return []
        candidates = sorted(candidates, key=lambda c: (-c[3], -c[2]))
        kept = []
        for x, y, r, score in candidates:
            duplicate = False
            for kx, ky, kr, _ in kept:
                center_dist = np.hypot(x - kx, y - ky)
                radius_ratio_diff = abs(r - kr) / max(r, kr)
                if (
                    center_dist < 0.70 * min(r, kr) and radius_ratio_diff < 0.45
                ) or overlap_ratio((x, y, r), (kx, ky, kr)) > 0.55:
                    duplicate = True
                    break
            if not duplicate:
                kept.append((x, y, r, score))
        return sorted(kept, key=lambda c: (c[1], c[0]))

    def has_nearby(circle, accepted):
        x, y, r = circle
        for ax, ay, ar, _ in accepted:
            d = np.hypot(x - ax, y - ay)
            if d < 0.80 * min(r, ar):
                return True
            if overlap_ratio((x, y, r), (ax, ay, ar)) > 0.40:
                return True
        return False

    h, w = preprocessed_image.shape[:2]
    edges = cv2.Canny(preprocessed_image, 55, 135)

    param2_values = [hough_param2 + 2, hough_param2, hough_param2 - 2, hough_param2 - 4]
    min_dist_values = [24, 18]

    hough_candidates = []
    for p2 in param2_values:
        if p2 < 18:
            continue
        for min_dist in min_dist_values:
            circles = cv2.HoughCircles(
                preprocessed_image,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=min_dist,
                param1=hough_param1,
                param2=p2,
                minRadius=min_radius,
                maxRadius=max_radius,
            )
            if circles is None:
                continue
            for x, y, r in np.uint16(np.around(circles))[0, :]:
                if x - r < 1 or y - r < 1 or x + r >= w - 1 or y + r >= h - 1:
                    continue
                score = edge_support_score(edges, int(x), int(y), int(r))
                coverage = angular_edge_coverage(edges, int(x), int(y), int(r))
                min_score = 0.10 if r <= 24 else 0.12
                min_coverage = 0.62 if r <= 24 else 0.68
                if score >= min_score and coverage >= min_coverage:
                    hough_candidates.append((int(x), int(y), int(r), score))

    accepted = suppress_duplicates(hough_candidates)

    # Fallback: only if likely under-detected.
    target_min_coins = 10
    if len(accepted) < target_min_coins:
        thresh = cv2.adaptiveThreshold(
            preprocessed_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < np.pi * (min_radius**2) * 0.55:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.75:
                continue
            (x_f, y_f), r_f = cv2.minEnclosingCircle(cnt)
            x, y, r = int(x_f), int(y_f), int(r_f)
            if r < min_radius or r > max_radius:
                continue
            if x - r < 1 or y - r < 1 or x + r >= w - 1 or y + r >= h - 1:
                continue
            score = edge_support_score(edges, x, y, r)
            coverage = angular_edge_coverage(edges, x, y, r)
            if score >= 0.11 and coverage >= 0.70:
                contour_candidates.append((x, y, r, score + 0.01))

        contour_candidates = suppress_duplicates(contour_candidates)
        for candidate in contour_candidates:
            if len(accepted) >= target_min_coins:
                break
            cx, cy, cr, cs = candidate
            if not has_nearby((cx, cy, cr), accepted):
                accepted.append((cx, cy, cr, cs))
        accepted = suppress_duplicates(accepted)

    if not accepted:
        return np.array([])
    return np.array([[x, y, r] for x, y, r, _ in accepted], dtype=np.uint16)


def extract_features(image, circles):
    if len(circles) == 0:
        return []
    radii = circles[:, 2].astype(float)
    min_r = float(np.min(radii))
    max_r = float(np.max(radii))

    coins = []
    for idx, (x, y, r) in enumerate(circles, start=1):
        color_name, color_hex = classify_coin_color(image, x, y, r)
        coins.append(
            {
                "id": idx,
                "center_x": int(x),
                "center_y": int(y),
                "radius_px": int(r),
                "diameter_px": int(2 * r),
                "area_px2": int(np.pi * r**2),
                "size": get_coin_size(r, min_r, max_r),
                "color": color_name,
                "sampled_color_hex": color_hex,
            }
        )
    return coins


def draw_results(image, coins):
    result = image.copy()
    for coin in coins:
        x, y, r = coin["center_x"], coin["center_y"], coin["radius_px"]
        cv2.circle(result, (x, y), r, (0, 255, 0), 2)
        cv2.circle(result, (x, y), 3, (0, 0, 255), -1)
        label = f"#{coin['id']}: {coin['size']}, {coin['color']}"
        cv2.putText(
            result,
            label,
            (x - 40, y - r - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
    return result


st.title("Coin Extractor UI")
st.write("Upload an image and extract coins similar to your notebook pipeline.")

with st.sidebar:
    st.header("Detection Settings")
    hough_param1 = st.slider("HOUGH_PARAM1", 30, 100, 50)
    hough_param2 = st.slider("HOUGH_PARAM2", 18, 50, 34)
    min_radius = st.slider("Min Radius", 5, 40, 10)
    max_radius = st.slider("Max Radius", 30, 200, 100)
    st.caption("Color classes: " + ", ".join(COIN_COLOR_PALETTE.keys()))

uploaded_file = st.file_uploader(
    "Choose a coin image", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    original, gray, blurred = preprocess_image(uploaded_file.read())
    circles = detect_coins(
        blurred,
        hough_param1=hough_param1,
        hough_param2=hough_param2,
        min_radius=min_radius,
        max_radius=max_radius,
    )
    coins = extract_features(original, circles)
    result = draw_results(original, coins)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), use_container_width=True)
    with c2:
        st.subheader("Detected Coins")
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.markdown(f"### Total Coins: {len(coins)}")

    if coins:
        df = pd.DataFrame(coins)
        st.dataframe(df, use_container_width=True)

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("Size Counts")
            size_counts = df["size"].value_counts().reset_index()
            size_counts.columns = ["size", "count"]
            st.bar_chart(size_counts.set_index("size"))
        with chart_col2:
            st.subheader("Color Counts")
            color_counts = df["color"].value_counts().reset_index()
            color_counts.columns = ["color", "count"]
            st.bar_chart(color_counts.set_index("color"))
    else:
        st.warning(
            "No coins detected. Try lowering HOUGH_PARAM2 or adjusting radius bounds."
        )
else:
    st.info("Upload an image to begin.")
