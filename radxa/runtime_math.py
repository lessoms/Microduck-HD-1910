import numpy as np

def gravity_from_quaternion(q):
    w,x,y,z=np.asarray(q,dtype=float)/np.linalg.norm(q)
    return np.array([2*(w*y-x*z),-2*(y*z+w*x),2*(x*x+y*y)-1])

def tilt_quaternion(g):
    # Shortest body-to-world tilt rotation. Heading is unobservable here.
    a=np.asarray(g);b=np.array([0.,0.,-1.]);dot=float(a@b)
    if dot<-.999999:return [0.,1.,0.,0.]
    q=np.r_[1+dot,np.cross(a,b)];return (q/np.linalg.norm(q)).tolist()
