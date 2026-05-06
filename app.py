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
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)
    return img, enhanced, blurred


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


COIN_COLOR_PROFILES = {
    "Silver": {
        "rgb": (192, 192, 192),
        "family": "neutral",
        "target_value": 185,
    },
    "Gold": {
        "rgb": (212, 175, 55),
        "family": "warm",
        "target_hue": 24,
        "target_saturation": 130,
        "target_value": 175,
    },
    "Copper": {
        "rgb": (184, 115, 51),
        "family": "warm",
        "target_hue": 10,
        "target_saturation": 150,
        "target_value": 145,
    },
    "Bronze": {
        "rgb": (150, 100, 45),
        "family": "warm",
        "target_hue": 18,
        "target_saturation": 120,
        "target_value": 115,
    },
    "Dark": {
        "rgb": (75, 75, 75),
        "family": "dark",
        "target_value": 65,
    },
}
COIN_COLOR_PALETTE = {
    name: profile["rgb"] for name, profile in COIN_COLOR_PROFILES.items()
}


def hue_distance(hue_a, hue_b):
    """Return circular OpenCV HSV hue distance on the 0..179 scale."""
    distance = abs(float(hue_a) - float(hue_b))
    return min(distance, 180.0 - distance)


def circular_median_hue(hues):
    """Estimate representative hue for sampled HSV pixels."""
    if len(hues) == 0:
        return 0.0
    radians = hues.astype(float) * 2.0 * np.pi / 180.0
    mean_angle = np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))
    return float((mean_angle * 180.0 / (2.0 * np.pi)) % 180.0)


def get_coin_color_sample(image, x, y, radius):
    """Sample stable interior coin pixels while preserving minority warm regions.

    Coin photos often contain strong white highlights, dark relief shadows, and
    gray-looking worn metal. A copper penny can therefore have a gray median even
    when a meaningful portion of its interior is orange/brown. This sampler keeps
    both all stable pixels and warm-colored pixels so scoring can distinguish
    silver coins from shiny copper coins.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    sample_radius = max(1, int(radius * 0.65))
    cv2.circle(mask, (int(x), int(y)), sample_radius, 255, -1)

    sampled_pixels = image[mask == 255]
    if sampled_pixels.size == 0:
        return None

    hsv_pixels = cv2.cvtColor(
        sampled_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)
    value_channel = hsv_pixels[:, 2]
    low_value, high_value = np.percentile(value_channel, [8, 92])
    stable_pixels = sampled_pixels[
        (value_channel >= low_value) & (value_channel <= high_value)
    ]
    if len(stable_pixels) < 20:
        stable_pixels = sampled_pixels

    stable_hsv = cv2.cvtColor(
        stable_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV
    ).reshape(-1, 3)
    stable_lab = cv2.cvtColor(
        stable_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3)

    hue = stable_hsv[:, 0]
    saturation = stable_hsv[:, 1]
    value = stable_hsv[:, 2]
    warm_mask = (saturation >= 24) & (value >= 45) & ((hue <= 32) | (hue >= 170))
    saturated_mask = saturation >= 28

    warm_hsv = stable_hsv[warm_mask]
    warm_lab = stable_lab[warm_mask]
    hue_source = warm_hsv if len(warm_hsv) >= 8 else stable_hsv[saturated_mask]
    if len(hue_source) == 0:
        hue_source = stable_hsv

    chroma_values = np.hypot(
        stable_lab[:, 1].astype(float) - 128.0, stable_lab[:, 2].astype(float) - 128.0
    )
    if len(warm_lab) > 0:
        warm_chroma_values = np.hypot(
            warm_lab[:, 1].astype(float) - 128.0,
            warm_lab[:, 2].astype(float) - 128.0,
        )
        warm_chroma = float(np.median(warm_chroma_values))
        warm_saturation = float(np.median(warm_hsv[:, 1]))
        warm_value = float(np.median(warm_hsv[:, 2]))
    else:
        warm_chroma = 0.0
        warm_saturation = 0.0
        warm_value = 0.0

    median_bgr = np.median(stable_pixels, axis=0).astype(np.uint8)
    median_rgb = median_bgr[::-1]

    return {
        "median_rgb": median_rgb,
        "median_hue": circular_median_hue(hue_source[:, 0]),
        "median_saturation": float(np.median(saturation)),
        "median_value": float(np.median(value)),
        "lab_chroma": float(np.median(chroma_values)),
        "warm_ratio": float(np.count_nonzero(warm_mask) / len(stable_hsv)),
        "warm_saturation": warm_saturation,
        "warm_value": warm_value,
        "warm_chroma": warm_chroma,
    }


def score_coin_color(sample, color_name):
    """Score how well a sampled coin matches a named color; lower is better."""
    profile = COIN_COLOR_PROFILES[color_name]
    saturation = sample["median_saturation"]
    value = sample["median_value"]
    chroma = sample["lab_chroma"]
    warm_ratio = sample["warm_ratio"]
    warm_saturation = sample["warm_saturation"]
    warm_value = sample["warm_value"]
    warm_chroma = sample["warm_chroma"]

    if profile["family"] == "dark":
        return (
            max(value - profile["target_value"], 0.0) / 45.0
            + saturation / 150.0
            + chroma / 45.0
            + warm_ratio * 5.0
        )

    if profile["family"] == "neutral":
        dark_penalty = max(95.0 - value, 0.0) / 20.0
        color_penalty = saturation / 55.0 + chroma / 25.0
        warm_penalty = warm_ratio * 9.0 + warm_saturation / 95.0 + warm_chroma / 28.0
        brightness_score = abs(value - profile["target_value"]) / 220.0
        return color_penalty + dark_penalty + warm_penalty + brightness_score

    hue_score = hue_distance(sample["median_hue"], profile["target_hue"]) / 10.0
    saturation_score = abs(warm_saturation - profile["target_saturation"]) / 155.0
    value_score = abs(warm_value - profile["target_value"]) / 180.0
    missing_warm_penalty = max(0.10 - warm_ratio, 0.0) * 35.0
    weak_warmth_penalty = max(18.0 - warm_chroma, 0.0) / 8.0
    dark_penalty = max(65.0 - value, 0.0) / 18.0
    return (
        hue_score
        + saturation_score
        + value_score
        + missing_warm_penalty
        + weak_warmth_penalty
        + dark_penalty
    )


def classify_coin_color(image, x, y, radius, palette=COIN_COLOR_PALETTE):
    """Classify a detected coin by choosing the closest allowed color.

    The classifier explicitly measures the ratio of orange/brown pixels inside
    the coin. That prevents copper pennies with gray highlights from being
    mislabeled as silver just because the overall median color is pale.
    """
    allowed_names = [name for name in palette if name in COIN_COLOR_PROFILES]
    if not allowed_names:
        return "Unknown", "#000000"

    sample = get_coin_color_sample(image, x, y, radius)
    if sample is None:
        return "Unknown", "#000000"

    best_name = min(allowed_names, key=lambda name: score_coin_color(sample, name))
    color_hex = "#{:02X}{:02X}{:02X}".format(*sample["median_rgb"])
    return best_name, color_hex


def detect_coins(
    preprocessed_image,
    hough_param1=50,
    hough_param2=34,
    min_radius=10,
    max_radius=100,
    use_contour_fallback=False,
):
    """Detect coin circles with validated Hough candidates and optional contour fallback."""

    def edge_support_score(edges, x, y, r):
        rim_mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.circle(rim_mask, (x, y), r, 255, 2)
        rim_pixels = np.count_nonzero(rim_mask)
        if rim_pixels == 0:
            return 0.0
        edge_pixels = np.count_nonzero(cv2.bitwise_and(edges, edges, mask=rim_mask))
        return edge_pixels / rim_pixels

    def angular_edge_metrics(edges, x, y, r, samples=96):
        """Measure rim continuity and the largest missing angular gap."""
        if r <= 0:
            return 0.0, 1.0

        h, w = edges.shape[:2]
        hits = []
        search_band = max(2, int(0.07 * r))

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
            hits.append(found)

        coverage = sum(hits) / float(samples)
        if not any(hits):
            return 0.0, 1.0

        doubled = hits + hits
        max_gap = 0
        current_gap = 0
        for hit in doubled:
            if hit:
                max_gap = max(max_gap, current_gap)
                current_gap = 0
            else:
                current_gap += 1
        max_gap = min(max(max_gap, current_gap), samples)
        return coverage, max_gap / float(samples)

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
                    center_dist < 0.75 * max(r, kr) and radius_ratio_diff < 0.55
                ) or overlap_ratio((x, y, r), (kx, ky, kr)) > 0.30:
                    duplicate = True
                    break
            if not duplicate:
                kept.append((x, y, r, score))
        return sorted(kept, key=lambda c: (c[1], c[0]))

    def has_nearby(circle, accepted):
        x, y, r = circle
        for ax, ay, ar, _ in accepted:
            d = np.hypot(x - ax, y - ay)
            if d < 0.85 * max(r, ar):
                return True
            if overlap_ratio((x, y, r), (ax, ay, ar)) > 0.25:
                return True
        return False

    h, w = preprocessed_image.shape[:2]
    edges = cv2.Canny(preprocessed_image, 45, 125)

    def valid_circle(x, y, r):
        return (
            min_radius <= r <= max_radius
            and x - r >= 1
            and y - r >= 1
            and x + r < w - 1
            and y + r < h - 1
        )

    def circle_quality(x, y, r):
        support = edge_support_score(edges, x, y, r)
        coverage, gap = angular_edge_metrics(edges, x, y, r)
        return support, coverage, gap, support + 0.35 * coverage - 0.20 * gap

    param2_values = sorted(
        {max(22, hough_param2 + offset) for offset in (6, 3, 0, -3, -6)},
        reverse=True,
    )
    min_dist_values = sorted(
        {max(12, int(min_radius * factor)) for factor in (1.6, 2.1, 2.8)},
        reverse=True,
    )

    hough_candidates = []
    for p2 in param2_values:
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
                x, y, r = int(x), int(y), int(r)
                if not valid_circle(x, y, r):
                    continue
                support, coverage, gap, quality = circle_quality(x, y, r)
                min_support = 0.11 if r <= 24 else 0.13
                min_coverage = 0.58 if r <= 24 else 0.64
                max_gap = 0.30 if r <= 24 else 0.25
                if (
                    support >= min_support
                    and coverage >= min_coverage
                    and gap <= max_gap
                ):
                    hough_candidates.append((x, y, r, quality))

    accepted = suppress_duplicates(hough_candidates)

    if use_contour_fallback:
        # Optional fallback for difficult photos. It is off by default because text,
        # logos, and watermarks can create circular-looking contours.
        contour_source = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
        )
        contours, _ = cv2.findContours(
            contour_source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < np.pi * (min_radius**2) * 0.60:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.72:
                continue
            (x_f, y_f), r_f = cv2.minEnclosingCircle(cnt)
            x, y, r = int(round(x_f)), int(round(y_f)), int(round(r_f))
            if not valid_circle(x, y, r):
                continue
            support, coverage, gap, quality = circle_quality(x, y, r)
            area_ratio = area / (np.pi * (r**2))
            if (
                0.50 <= area_ratio <= 1.10
                and support >= 0.12
                and coverage >= 0.62
                and gap <= 0.28
            ):
                contour_candidates.append((x, y, r, quality + circularity * 0.05))

        for candidate in suppress_duplicates(contour_candidates):
            cx, cy, cr, _ = candidate
            if not has_nearby((cx, cy, cr), accepted):
                accepted.append(candidate)
        accepted = suppress_duplicates(accepted)

    if not accepted:
        return np.array([])
    return np.array([[x, y, r] for x, y, r, _ in accepted], dtype=np.uint16)


def extract_features(image, circles, palette=COIN_COLOR_PALETTE):
    if len(circles) == 0:
        return []
    radii = circles[:, 2].astype(float)
    min_r = float(np.min(radii))
    max_r = float(np.max(radii))

    coins = []
    for idx, (x, y, r) in enumerate(circles, start=1):
        color_name, color_hex = classify_coin_color(image, x, y, r, palette)
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
    hough_param2 = st.slider("HOUGH_PARAM2", 22, 50, 34)
    min_radius = st.slider("Min Radius", 5, 40, 10)
    max_radius = st.slider("Max Radius", 30, 200, 100)
    use_contour_fallback = st.checkbox(
        "Enable contour fallback",
        value=False,
        help="Try this only if Hough misses coins; it can detect logos/text as coins.",
    )
    selected_color_names = st.multiselect(
        "Color classes",
        options=list(COIN_COLOR_PALETTE.keys()),
        default=list(COIN_COLOR_PALETTE.keys()),
        help="Only these labels will be used when assigning each detected coin color.",
    )
    selected_color_palette = {
        name: COIN_COLOR_PALETTE[name] for name in selected_color_names
    }

uploaded_file = st.file_uploader(
    "Choose a coin image", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    original, _enhanced, blurred = preprocess_image(uploaded_file.read())
    circles = detect_coins(
        blurred,
        hough_param1=hough_param1,
        hough_param2=hough_param2,
        min_radius=min_radius,
        max_radius=max_radius,
        use_contour_fallback=use_contour_fallback,
    )
    coins = extract_features(original, circles, selected_color_palette)
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
