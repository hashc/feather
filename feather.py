# -*- coding: utf-8 -*-
import numpy as np
from scipy import interpolate

def sampleCubicSplinesWithDerivative(points, tangents, resolution):
    resolution = float(resolution)
    points = np.asarray(points)
    nPoints, dim = points.shape

    # Parametrization parameter s.
    dp = np.diff(points, axis=0)                 # difference between points
    dp = np.linalg.norm(dp, axis=1)              # distance between points
    d = np.cumsum(dp)                            # cumsum along the segments
    d = np.hstack([[0],d])                       # add distance from first point
    l = d[-1]                                    # length of point sequence
    nSamples = int(l/resolution)                 # number of samples
    s,r = np.linspace(0,l,nSamples,retstep=True) # sample parameter and step

    # Bring points and (optional) tangent information into correct format.
    assert(len(points) == len(tangents))
    data = np.empty([nPoints, dim], dtype=object)
    for i,p in enumerate(points):
        t = tangents[i]
        # Either tangent is None or has the same
        # number of dimensions as the point p.
        assert(t is None or len(t)==dim)
        fuse = list(zip(p,t) if t is not None else zip(p,))
        data[i,:] = fuse

    # Compute splines per dimension separately.
    samples = np.zeros([nSamples, dim])
    poly={}
    for i in range(dim):
        poly[i] = interpolate.BPoly.from_derivatives(d, data[:,i])
        samples[:,i] = poly[i](s)
    return samples


def curverfit(p,r,resolution = 0.1):
    points = []
    tangents = []
    for i in range(len(p)):
        points.append(p[i]); tangents.append(r[i])
    
    points = np.asarray(points)
    tangents = np.asarray(tangents)
    
    scale = 0.1
    tangents1 = np.dot(tangents, scale*np.eye(3))
    samples1 = sampleCubicSplinesWithDerivative(points, tangents1, resolution)
    return samples1

def CubicHermiteCurvel(p,r,num):
    c=[]
    for i in range(3):
        c.append([p[0,i],r[0,i],-3*p[0,i]+3*p[1,i]-2*r[0,i]-r[1,i],2*p[0,i]-2*p[1,i]+r[0,i]+r[1,i]])
    c=np.array(c)
    v=[]
    v.append(list(p[0]))
    t=0
    delta=1/num
    for i in range(1,num):
        t+=delta
        temp= list(c[:,0]+t*(c[:,1]+t*(c[:,2]+t*c[:,3])))
        v.append(temp)
    return np.array(v),c





import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def givename(k):
    string=str(k)
    n=len(string)
    if n<5:
        string='0'*(5-n)+string
    else:
        pass
    return string


def myfun(i):
    delta=[i,i]
    return delta
def animate(n):

    fig = plt.figure(figsize=(12,12),facecolor='snow')
    ax = fig.add_subplot(111, projection='3d')
    # 主干
    p=np.array([[0,4,0],
                [0,-3,0]])
    r=np.array([[1,-3,0],
                [3,-6,0]])
    #samples1= curverfit(p,r,resolution = 0.1)
    samples1,c1=CubicHermiteCurvel(p,r,100)
    ax.plot(samples1[:,0], samples1[:,1],samples1[:,2], c=(0.6,0.6,0.6),linestyle='solid',linewidth=1)


    # 主羽枝
    ww=0.01
    t9=np.linspace(0.01,0.3,30)
    for i in t9:
        #x9,y9,z9=poly[0](i),poly[1](i),poly[2](i)
        x9,y9,_=list(c1[:,0]+i*(c1[:,1]+i*(c1[:,2]+i*c1[:,3])))
        x19=1/6-np.sqrt((1/12+4-y9-x9)/3)
        y19=y9+x9-x19+i*0.1
        d9=np.sqrt(2)*abs(x19-x9)
        p9=np.array([[x9,y9,0],[x19,y19,0]])
        r9=np.array([[-1*d9,0.5*d9,0],[-1*d9,2*d9,0]])
        samples9,c9 = CubicHermiteCurvel(p9,r9,80)
        ax.plot(samples9[:,0], samples9[:,1],samples9[:,2],c=(0.6,0.6,0.6),linestyle='solid',linewidth=0.5+3*i)


    #左羽
    t=np.linspace(0.303,0.72,140)
    Max_random=1000
    for i in t:
        x,y,_=list(c1[:,0]+i*(c1[:,1]+i*(c1[:,2]+i*c1[:,3])))
        w2=np.random.rand()
        xl=1/6-np.sqrt((1/12+4-y-x)/3)-0.1+0.3*w2
        yl=y+x-xl+i*0.1
        d=np.sqrt(2)*abs(xl-x)
        p=np.array([[x, y, 0.0],[xl, yl, 0]])
        ww2=np.random.rand()/Max_random
        www2=np.random.rand()/Max_random
        r=np.array([[-1.0*d*np.cos (5.0*np.pi/6.0+3.0*ww2*np.pi/4.0),
                    0.5*d*np.sin(5.0*np.pi/6.0+3*ww2*np.pi/4.0),0.0],
                        [-1.0*d*np.cos(-5*np.pi/4+3*www2*np.pi/2),
                        2*d*np.sin(-5*np.pi/4+3*www2*np.pi/2),0]])
        samples2,c2 = CubicHermiteCurvel(p,r,80)
        ax.plot(samples2[:,0], samples2[:,1],samples2[:,2],c=(0.6,0.6,0.6),linestyle='solid',linewidth=0.5)

    tl=np.linspace(0.726,0.8,30)
    for i in tl:
        xx,yy,_=list(c1[:,0]+i*(c1[:,1]+i*(c1[:,2]+i*c1[:,3])))
        w3=np.random.rand()/Max_random
        xxl=xl+0.008+20.0*(i-0.72)*(i-0.72)-0.1+0.3*w3;
        yyl=yy+xx-xxl+i*0.1-(i-0.72)*1.8;
        d=np.sqrt(2)*abs(xxl-xx)
        p2=np.array([[xx, yy, 0.0],[xxl, yyl, 0]])
        ww3=np.random.rand()/Max_random
        www3=np.random.rand()/Max_random
        r2=np.array([[-1.0*d*np.cos(5.0*np.pi/6.0+3.0*ww3*np.pi/4.0),
                      0.5*d*np.sin(5.0*np.pi/6.0+3.0*ww3*np.pi/4.0),0.0],
                      [-1.0*d*np.cos(-5.0*np.pi/4.0+3.0*www3*np.pi/2.0),
                        2.0*d*np.sin(-5.0*np.pi/4.0+3.0*www3*np.pi/2.0),0.0]])
        samples3,c3 = CubicHermiteCurvel(p2,r2,80)
        ax.plot(samples3[:,0], samples3[:,1],samples3[:,2],c=(0.6,0.6,0.6),linestyle='solid',linewidth=0.5)
    ax.set_xlim(-5,5)
    ax.set_axis_off()
    a,b=myfun(n)
    ax.view_init(a,b)
    plt.savefig('/mnt/d/Research/feather/res/%s.png'%(givename(n)))
    plt.clf()
    plt.cla()
    return 0
import os
if not os.path.exists('./res'):
    os.mkdir('./res')
for i in range(180):
    animate(i)

os.system('convert -delay 24 -loop 0 /mnt/d/Research/feather/res/*.png  /mnt/d/Research/feather/out.gif') 