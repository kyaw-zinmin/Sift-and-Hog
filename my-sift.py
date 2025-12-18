import cv2
import matplotlib.pyplot as plt
import numpy as np

# helper functions
# get hr corner image, and hr dst
def hr_corners(original_image, gray_image, get_dst=False):
    dst = cv2.cornerHarris(gray_image, 5, 3, 0.02)
    dst = cv2.dilate(dst, None)

    thresh = 0.01 * dst.max()
    corners = np.argwhere(dst > thresh)
    for y, x in corners:
        cv2.circle(original_image, (x, y), radius=10, color=(0, 0, 255), thickness=-1)

    if (get_dst == True):
        return original_image, dst
    else:
        return original_image
    
# convert hr dst to sift key poitns
def hr_dst_to_sift_kp(hr_dst):
    threshold = 0.01 * hr_dst.max()
    corner = np.where(hr_dst > threshold)
    
    sift_kp = []
    for y, x in zip(corner[0], corner[1]):
        kp = cv2.KeyPoint(
            x=float(x), y=float(y),
            # adjust this value to resize the sift keypoint circle size 
            size=50
        )
        sift_kp.append(kp)
    return sift_kp

# get sift descriptor image, and sift key points and descriptors
def sift_at_hr(original_image, gray_image, hr_kp, get_sift=False):
    sift = cv2.SIFT.create()
    sift_kp, sift_des = sift.compute(gray_image, hr_kp)

    sift_at_hr_img = cv2.drawKeypoints(
        gray_image, sift_kp, original_image,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    if (get_sift == True):
        return sift_at_hr_img, sift_kp, sift_des
    else:
        return sift_at_hr_img
    
def main():
    # Terminology

    # ref = the reference image
    # tar = the target image, the one we are trying to match
    # hr = Harris / Harris Corner

    # Important variables
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    ref = cv2.imread('./my-sift-images/reference.jpg')
    tar = cv2.imread('./my-sift-images/target.jpg')

    # Empty image variables

    # Harris Corner points image
    ref_hr_corners_img = None
    tar_hr_corners_img = None
    # Sift descriptors at Harris Corner points image
    ref_sift_at_hr_img = None
    tar_sift_at_hr_img = None
    # Sift matched image
    sift_matched = None

    # transform images to gray
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    tar_gray = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)

    # corner image and harris destinations are obtained here
    ref_hr_corners_img, dst_ref = hr_corners(ref, ref_gray, True)
    tar_hr_corners_img, dst_tar = hr_corners(tar, tar_gray, True)

    # convert hr dst to sift key points
    ref_hr_kp = hr_dst_to_sift_kp(dst_ref)
    tar_hr_kp = hr_dst_to_sift_kp(dst_tar)
    
    # get sift descriptor image, sift keypoint, sift descriptor
    ref_sift_at_hr_img, ref_sift_kp, ref_sift_des = sift_at_hr(ref, ref_gray, ref_hr_kp, True)
    tar_sift_at_hr_img, tar_sift_kp, tar_sift_des = sift_at_hr(tar, tar_gray, tar_hr_kp, True)

    # match the descriptors and get matched image
    matches = matcher.match(ref_sift_des, tar_sift_des)
    matches = sorted(matches, key=lambda x: x.distance)
    sift_matched = cv2.drawMatches(
        ref, ref_sift_kp,
        tar, tar_sift_kp,
        matches[:50],
        tar, flags=2,
        matchesThickness=4
    )

    cv2.imshow("Harris Corner at Reference", ref_hr_corners_img)
    cv2.imshow("Harris Corner at Target", tar_hr_corners_img)
    cv2.imshow("Sift Descriptor at Reference", ref_sift_at_hr_img)
    cv2.imshow("Sift Descriptor at Target", tar_sift_at_hr_img)
    cv2.imshow("SIFT Matched Image", sift_matched)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    cv2.waitKey(0)

main()