# detect_lines_hough.py

"""
Detect straight lines using the standard Hough Transform (cv2.HoughLines)
and draw them on the image.
"""
import cv2
import numpy as np
import argparse

def detect_and_draw_lines(image_path, output_path=None,
                          canny_thresh1=50, canny_thresh2=150,
                          rho=1, theta=np.pi/180, threshold=200):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    img_draw = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2, apertureSize=3)

    lines = cv2.HoughLines(edges, rho, theta, threshold)
    if lines is not None:
        for rho_i, theta_i in lines[:,0]:
            a = np.cos(theta_i)
            b = np.sin(theta_i)
            x0 = a * rho_i
            y0 = b * rho_i
            scale = max(img.shape[:2])
            x1 = int(x0 + scale * (-b))
            y1 = int(y0 + scale * ( a))
            x2 = int(x0 - scale * (-b))
            y2 = int(y0 - scale * ( a))
            cv2.line(img_draw, (x1,y1), (x2,y2), (0,0,255), 2, cv2.LINE_AA)
    else:
        print("No lines detected.")

    cv2.imshow("Original", img)
    cv2.imshow("Edges", edges)
    cv2.imshow("HoughLines", img_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output_path:
        cv2.imwrite(output_path, img_draw)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('-o','--output', help='Save annotated image')
    parser.add_argument('--canny1', type=int, default=50)
    parser.add_argument('--canny2', type=int, default=150)
    parser.add_argument('--rho', type=float, default=1.0)
    parser.add_argument('--theta', type=float, default=np.pi/180)
    parser.add_argument('--threshold', type=int, default=200)
    args = parser.parse_args()
    detect_and_draw_lines(
        args.image, args.output,
        args.canny1, args.canny2,
        args.rho, args.theta, args.threshold
    )


# detect_lines_houghp.py

"""
Detect straight line segments using Probabilistic Hough Transform (cv2.HoughLinesP)
and draw them on the image.
"""
import cv2
import numpy as np
import argparse

def detect_and_draw_probabilistic_lines(
    image_path, output_path=None,
    canny_thresh1=50, canny_thresh2=150,
    rho=1.0, theta=np.pi/180, threshold=50,
    min_line_length=50, max_line_gap=10
):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    drawn = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)

    lines = cv2.HoughLinesP(
        edges, rho, theta, threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            cv2.line(drawn, (x1,y1), (x2,y2), (0,255,0), 2, cv2.LINE_AA)
    else:
        print("No segments detected.")

    cv2.imshow("Original", img)
    cv2.imshow("Edges", edges)
    cv2.imshow("HoughLinesP", drawn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output_path:
        cv2.imwrite(output_path, drawn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('-o','--output')
    parser.add_argument('--canny1', type=int, default=50)
    parser.add_argument('--canny2', type=int, default=150)
    parser.add_argument('--rho', type=float, default=1.0)
    parser.add_argument('--theta', type=float, default=np.pi/180)
    parser.add_argument('--threshold', type=int, default=50)
    parser.add_argument('--minlen', type=int, default=50)
    parser.add_argument('--maxgap', type=int, default=10)
    args = parser.parse_args()
    detect_and_draw_probabilistic_lines(
        args.image, args.output,
        args.canny1, args.canny2,
        args.rho, args.theta, args.threshold,
        args.minlen, args.maxgap
    )


# detect_circles_hough.py

"""
Detect circles using Hough Circle Transform (cv2.HoughCircles) and visualize them.
"""
import cv2
import numpy as np
import argparse

def detect_and_draw_circles(
    image_path, output_path=None,
    dp=1.2, min_dist=20,
    param1=50, param2=30,
    min_radius=0, max_radius=0
):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    output = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp, min_dist,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for idx, (x,y,r) in enumerate(circles[0,:], start=1):
            cv2.circle(output, (x,y), r, (0,255,0), 2)
            cv2.circle(output, (x,y), 2, (0,0,255), 3)
            cv2.putText(output, str(idx), (x-10,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    else:
        print("No circles detected.")

    cv2.imshow("Circles", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output_path:
        cv2.imwrite(output_path, output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('-o','--output')
    parser.add_argument('--dp', type=float, default=1.2)
    parser.add_argument('--min_dist', type=float, default=20)
    parser.add_argument('--param1', type=float, default=50)
    parser.add_argument('--param2', type=float, default=30)
    parser.add_argument('--min_radius', type=int, default=0)
    parser.add_argument('--max_radius', type=int, default=0)
    args = parser.parse_args()
    detect_and_draw_circles(
        args.image, args.output,
        dp=args.dp, min_dist=args.min_dist,
        param1=args.param1, param2=args.param2,
        min_radius=args.min_radius, max_radius=args.max_radius
    )


# hough_trackbar.py

"""
Create a trackbar GUI to tune HoughLinesP parameters in real time.
"""
import cv2
import numpy as np
import argparse

def on_trackbar(val):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("Controls")
    cv2.createTrackbar("Canny Th1","Controls",50,500,on_trackbar)
    cv2.createTrackbar("Canny Th2","Controls",150,500,on_trackbar)
    cv2.createTrackbar("Thresh","Controls",50,500,on_trackbar)
    cv2.createTrackbar("MinLen","Controls",50,500,on_trackbar)
    cv2.createTrackbar("MaxGap","Controls",10,500,on_trackbar)

    while True:
        t1 = cv2.getTrackbarPos("Canny Th1","Controls")
        t2 = cv2.getTrackbarPos("Canny Th2","Controls")
        thresh = cv2.getTrackbarPos("Thresh","Controls")
        minLen = cv2.getTrackbarPos("MinLen","Controls")
        maxGap = cv2.getTrackbarPos("MaxGap","Controls")

        edges = cv2.Canny(gray, t1, t2)
        lines = cv2.HoughLinesP(edges,1,np.pi/180,thresh,minLen,maxGap)

        result = img.copy()
        if lines is not None:
            for x1,y1,x2,y2 in lines[:,0]:
                cv2.line(result,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.imshow("Edges", edges)
        cv2.imshow("Result", result)
        if cv2.waitKey(100)&0xFF in (27,ord('q')):
            break

    cv2.destroyAllWindows()

if __name__=='__main__':
    main()


# lane_detection_hough.py

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def draw_lines(img, lines, color=(0,255,0), thickness=5):
    if lines is None: return
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img,(x1,y1),(x2,y2),color,thickness)

import cv2
import numpy as np
import argparse

def process_video(input_path, output_path=None,
                  canny1=50, canny2=150,
                  rho=2, theta=np.pi/180, threshold=100,
                  min_len=40, max_gap=5):
    cap = cv2.VideoCapture(input_path)
    writer=None
    if output_path:
        fcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path,fcc,fps,(w,h))
    while True:
        ret,frame=cap.read()
        if not ret: break
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur=cv2.GaussianBlur(gray,(5,5),0)
        edges=cv2.Canny(blur,canny1,canny2)
        h,w=edges.shape
        verts=np.array([[ (w*0.1,h),(w*0.45,h*0.6),(w*0.55,h*0.6),(w*0.9,h) ]],np.int32)
        masked=region_of_interest(edges,verts)
        lines=cv2.HoughLinesP(masked,rho,theta,threshold,minLineLength=min_len,maxLineGap=max_gap)
        line_img=np.zeros_like(frame)
        draw_lines(line_img,lines)
        combo=cv2.addWeighted(frame,0.8,line_img,1.0,0)
        cv2.imshow('Lane Detection',combo)
        if writer: writer.write(combo)
        if cv2.waitKey(1)&0xFF in (27,ord('q')): break
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('input_video')
    parser.add_argument('-o','--output')
    parser.add_argument('--canny1',type=int,default=50)
    parser.add_argument('--canny2',type=int,default=150)
    parser.add_argument('--rho',type=float,default=2)
    parser.add_argument('--theta',type=float,default=np.pi/180)
    parser.add_argument('--threshold',type=int,default=100)
    parser.add_argument('--minlen',type=int,default=40)
    parser.add_argument('--maxgap',type=int,default=5)
    args=parser.parse_args()
    process_video(
        args.input_video, args.output,
        args.canny1, args.canny2,
        args.rho, args.theta, args.threshold,
        args.minlen, args.maxgap
    )

# canny_hough_demo.py
import cv2
import numpy as np
import argparse

def detect_lines_standard(img, edges, rho, theta, threshold):
    lines=cv2.HoughLines(edges, rho,theta,threshold)
    out=img.copy()
    if lines is not None:
        for rho_i,theta_i in lines[:,0]:
            a=np.cos(theta_i); b=np.sin(theta_i)
            x0=a*rho_i; y0=b*rho_i
            scale=max(img.shape[:2])
            x1=int(x0+scale*(-b)); y1=int(y0+scale*(a))
            x2=int(x0-scale*(-b)); y2=int(y0-scale*(a))
            cv2.line(out,(x1,y1),(x2,y2),(0,0,255),2,cv2.LINE_AA)
    return out

def detect_lines_probabilistic(img, edges, rho,theta, threshold, minlen, maxgap):
    lines=cv2.HoughLinesP(edges,rho,theta,threshold,minLineLength=minlen,maxLineGap=maxgap)
    out=img.copy()
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            cv2.line(out,(x1,y1),(x2,y2),(0,255,0),2,cv2.LINE_AA)
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--canny1', type=int, default=50)
    parser.add_argument('--canny2', type=int, default=150)
    parser.add_argument('--rho', type=float, default=1.0)
    parser.add_argument('--theta', type=float, default=np.pi/180)
    parser.add_argument('--threshold', type=int, default=100)
    parser.add_argument('--minlen', type=int, default=50)
    parser.add_argument('--maxgap', type=int, default=10)
    args=parser.parse_args()

    img=cv2.imread(args.image)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,args.canny1,args.canny2)

    standard=detect_lines_standard(img,edges,args.rho,args.theta,args.threshold)
    probabilistic=detect_lines_probabilistic(img,edges,args.rho,args.theta,args.threshold,args.minlen,args.maxgap)

    cv2.imshow('Original',img)
    cv2.imshow('Edges',edges)
    cv2.imshow('Standard Hough',standard)
    cv2.imshow('Probabilistic Hough',probabilistic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
