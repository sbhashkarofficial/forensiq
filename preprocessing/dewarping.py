import cv2
import numpy as np


def order_points(pts):
    """Sort 4 corner points in order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect


def unwarp_quadrilateral(image, src_pts):
    """Unwarps a 4-point quadrilateral into a flat, parallel-aligned rectangle."""
    src_pts = order_points(src_pts)
    tl, tr, br, bl = src_pts

    width_A = np.linalg.norm(br - bl)
    width_B = np.linalg.norm(tr - tl)
    max_width = max(int(width_A), int(width_B))

    height_A = np.linalg.norm(tr - br)
    height_B = np.linalg.norm(tl - bl)
    max_height = max(int(height_A), int(height_B))

    # Standard axis-aligned parallel target rectangle
    dst_pts = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped, max_width, max_height


def calculate_vertex_angle(prev_pt, curr_pt, next_pt):
    """Calculates internal angle (degrees) at curr_pt."""
    v1 = prev_pt - curr_pt
    v2 = next_pt - curr_pt

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))


def dewarp_image(image_path, contour, output_path="arap_flattened.jpg"):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")

    peri = cv2.arcLength(contour, True)

    # Prefer 6 edges over 4
    approx = None
    for eps_factor in [0.01, 0.015, 0.02, 0.03]:
        candidate = cv2.approxPolyDP(contour, eps_factor * peri, True)
        if len(candidate) == 6:
            approx = candidate
            break
        elif len(candidate) == 4 and approx is None:
            approx = candidate

    if approx is None:
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    pts = approx.reshape(-1, 2)
    num_edges = len(pts)

    if num_edges == 6:
        # Step 1: Find spine vertices via largest interior angles
        angles = []
        for i in range(6):
            prev_pt = pts[i - 1]
            curr_pt = pts[i]
            next_pt = pts[(i + 1) % 6]
            angles.append(calculate_vertex_angle(prev_pt, curr_pt, next_pt))

        spine_indices = np.argsort(angles)[-2:]
        outer_indices = [i for i in range(6) if i not in spine_indices]

        spine_pts = pts[spine_indices]
        outer_pts = pts[outer_indices]

        # Step 2: Determine if spine is vertical or horizontal
        dx = abs(spine_pts[0][0] - spine_pts[1][0])
        dy = abs(spine_pts[0][1] - spine_pts[1][1])
        is_vertical_spine = dy > dx

        part1_outer, part2_outer = [], []

        if is_vertical_spine:
            # Sort spine top-to-bottom
            if spine_pts[0][1] > spine_pts[1][1]:
                spine_pts = spine_pts[::-1]

            spine_vec = spine_pts[1] - spine_pts[0]
            for pt in outer_pts:
                pt_vec = pt - spine_pts[0]
                cross_product = spine_vec[0] * pt_vec[1] - spine_vec[1] * pt_vec[0]
                if cross_product > 0:
                    part2_outer.append(pt)  # Right
                else:
                    part1_outer.append(pt)  # Left
        else:
            # Sort spine left-to-right
            if spine_pts[0][0] > spine_pts[1][0]:
                spine_pts = spine_pts[::-1]

            spine_vec = spine_pts[1] - spine_pts[0]
            for pt in outer_pts:
                pt_vec = pt - spine_pts[0]
                cross_product = spine_vec[0] * pt_vec[1] - spine_vec[1] * pt_vec[0]
                if cross_product > 0:
                    part2_outer.append(pt)  # Bottom
                else:
                    part1_outer.append(pt)  # Top

        # Fallback split if cross product fails on unusual rotations
        if len(part1_outer) != 2 or len(part2_outer) != 2:
            centroid = np.mean(pts, axis=0)
            if is_vertical_spine:
                part1_outer = [pt for pt in outer_pts if pt[0] < centroid[0]]
                part2_outer = [pt for pt in outer_pts if pt[0] >= centroid[0]]
            else:
                part1_outer = [pt for pt in outer_pts if pt[1] < centroid[1]]
                part2_outer = [pt for pt in outer_pts if pt[1] >= centroid[1]]

        # Step 3: Unwarp quadrilaterals
        quad1_pts = np.vstack([part1_outer, spine_pts]).astype("float32")
        quad2_pts = np.vstack([part2_outer, spine_pts]).astype("float32")

        warp1, w1, h1 = unwarp_quadrilateral(image, quad1_pts)
        warp2, w2, h2 = unwarp_quadrilateral(image, quad2_pts)

        # Step 4: Align parallel dimensions and stitch along the spine
        if is_vertical_spine:
            # Match heights for horizontal stitching (Left + Right)
            target_h = max(h1, h2)
            warp1 = cv2.resize(warp1, (w1, target_h))
            warp2 = cv2.resize(warp2, (w2, target_h))
            unwarped_image = np.hstack((warp1, warp2))
        else:
            # Match widths for vertical stitching (Top + Bottom)
            target_w = max(w1, w2)
            warp1 = cv2.resize(warp1, (target_w, h1))
            warp2 = cv2.resize(warp2, (target_w, h2))
            unwarped_image = np.vstack((warp1, warp2))

    elif num_edges == 4:
        unwarped_image, _, _ = unwarp_quadrilateral(image, pts.astype("float32"))

    else:
        x, y, w, h = cv2.boundingRect(contour)
        unwarped_image = image[y : y + h, x : x + w]

    cv2.imwrite(output_path, unwarped_image)
    return unwarped_image