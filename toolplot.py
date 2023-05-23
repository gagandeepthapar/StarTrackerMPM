import os
import constants as c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.stats import halfnorm, norm

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str)
    parser.add_argument('-s', type=str)
    parser.add_argument('-p', action='store_true')

    return parser.parse_args()

def mcplot():

    return

if __name__ == '__main__':




    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # x, y = np.random.rand(2, 100) * 4
    # print(x)
    # hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])

    # # Construct arrays for the anchor positions of the 16 bars.
    # xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")

    # print(xpos, ypos, hist)

    # xpos = xpos.ravel()
    # ypos = ypos.ravel()
    # zpos = 0

    # # Construct arrays with the dimensions for the 16 bars.
    # dx = dy = 0.5 * np.ones_like(zpos)
    # dz = hist.ravel()

    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

    # plt.show()

    # raise ValueError
    args = parse_args()
    fname = os.path.join(c.curFile,args.f)
    plt.rcParams['text.usetex'] = True


    titles = {
        'BASE': 'Ideal Star Tracker',
        'CENTROID': 'Effects of Centroiding Accuracy',
        'EPS': 'Effects of 3DOF Image Plane Translation',
        'EPS_LAT': 'Effects of Lateral Image Plane Translation',
        'EPS_LONG': 'Effects of Vertical Image Plane Translation',
        'FOCLEN': 'Effects of Focal Length (FOV)',
        'FULL': 'Aggregate of all Effects',
        'FULLBETTE': 'Aggregate of all Effects',
        'PHI': 'Effects of 3DOF Image Plane Rotation',
        'PHI_LAT': 'Effects of Lateral Image Plane Rotation',
        'PHI_LONG': 'Effects of Vertical Image Plane Rotation',
        'STARS': 'Effects of Maximum Star Visibility',
    }
    units = {
        'CENTROID': 'Centroiding Accuracy [px]',
        'EPS': 'Total Image Plane Translation [px]',
        'EPS_LAT': 'Lateral Image Plane Translation [px]',
        'EPS_LONG': 'Vertical Image Plane Translation [px]',
        'FOCLEN': 'Focal Length (FOV) [px]',
        'PHI': 'Total Image Plane Rotation [deg]',
        'PHI_LAT': 'Image Plane Rotation about Lateral Axes [deg]',
        'PHI_LONG': 'Image Plane Rotation about Vertical Axis [deg]',
        'STARS': 'Maximum Visible Star Magnitude', 
    }
    sensetitle = {
        'CENTROID': 'Estimated Accuracy VS Centroiding Accuracy',
        'EPS': 'Estimated Accuracy VS Total Image Plane Translation',
        'EPS_LAT': 'Estimated Accuracy VS Lateral Image Plane Translation',
        'EPS_LONG': 'Estimated Accuracy VS Vertical Image Plane Translation',
        'FOCLEN': 'Estimated Accuracy VS Focal Length',
        'PHI': 'Estimated Accuracy VS Total Image Plane Rotation',
        'PHI_LAT': 'Estimated Accuracy VS Lateral Image Plane Rotation',
        'PHI_LONG': 'Estimated Accuracy VS Vertical Image Plane Rotation',
        'STARS': 'Estimated Accuracy VS Maximum Visible Magnitude',
    }


    colname = "QUATERNION_ERROR_[arcsec]"
    pname = args.f.split('/')[1].split('.')[0]

    mult = 1/(1-2/np.pi)**(5/10)
    if pname == 'EPS_LONG':
        mult = np.sqrt(2/np.pi)
    elif pname == 'SUPERCENTROID':
        mult = 1/(2/np.pi)**(5/10)

    half_normal_std = lambda x: x * mult
    half_normal_mean = lambda x: half_normal_std(x) * mult

    datadf:pd.DataFrame = pd.read_pickle(fname)
    # print(half_normal_mean(datadf[colname].std()))
    # raise ValueError

    bins = 50
    halfnormal = lambda x, s: np.sqrt(2)/(s*np.sqrt(np.pi)) * np.exp(-x**2 / (2*s**2))

    print(datadf[colname].std())
    datadf = datadf[datadf[colname] <= 6*datadf[colname].std()]
    # raise ValueError

    if args.s == 'm':
        x_data = np.linspace(0, 1.1*datadf[colname].max(), 100)
        hn_sig = half_normal_std(datadf[colname].std())

        halfdata_div = halfnormal(x_data, hn_sig)

        # datadf = datadf[datadf[colname] <= 3*hn_sig]

        fig, ax = plt.subplots()
        ax.hist(datadf[colname], bins=bins, density=True, label='Simulation Data')
        if hn_sig > 1e-5:
            label=r'Half-Normal Distribution, $\sigma$='+f'{np.round(hn_sig,3)}'
            ax.plot(x_data, halfdata_div, 'r', label=label)
            ax.legend(fontsize=12)
        # ax.set_title(r'\begin{center}'+f'{titles[pname]}\n'+r'$\pm$'+f'{hn_sig:.3}'+r' Arcsec ($1\sigma$)\end{center}', fontsize=15)
        ax.set_xlabel(colname.replace('_',' ').title(), fontsize=15)
        ax.set_ylabel('Probability Density', fontsize=15)

        ttil = '/Users/gagandeepthapar/Desktop/School/AERO/MS/Thesis/Documents/StarTrackerThesis/chapters/5_Univariate_Effect_Analysis/images/' +pname+f'_{int(hn_sig*1000)}'+'.png'

        plt.subplots_adjust(top=0.96, left=.11, right=.957)
        
        if args.p:
            plt.show()
        else:
            plt.savefig(ttil, dpi=600)
        # plt.show()

    if args.s == 's':
        PER_PARAM = 20
        params = []
        paramname = pname.split('_')[:-1]
        paramname = '_'.join(paramname)
        match paramname:
            case 'CENTROID':
                params = ['BASE_DEV_X', 'BASE_DEV_Y']
            
            case 'EPS':
                params = ['F_ARR_EPS_X', 'F_ARR_EPS_Y', 'F_ARR_EPS_Z']

            case 'EPS_LAT':
                params = ['F_ARR_EPS_X', 'F_ARR_EPS_Y']
            
            case 'EPS_LONG':
                params = ['F_ARR_EPS_Z']

            case 'FOCLEN':
                params = ['FOCAL_LENGTH']
            
            case 'PHI':
                params = ['F_ARR_PHI', 'F_ARR_THETA', 'F_ARR_PSI']
            
            case 'PHI_LAT':
                params = ['F_ARR_PHI', 'F_ARR_THETA']
            
            case 'PHI_LONG':
                params = ['F_ARR_PSI']
            
            case 'STARS':
                params = ['MAX_MAGNITUDE']



        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')

        fullparams = [*params, colname]
        data = datadf[fullparams].copy()

        data['MAG'] = data[params].apply(lambda x: np.linalg.norm(x)*np.sign(x[0]), axis=1)
        data = data[data[colname] > 0]
        # print(data[colname].max())
        # raise ValueError

        fig = plt.figure()
        
        bincount = 20
        ax = fig.add_subplot()
        xdata = data.MAG
        # print(xdata.min())
        # raise ValueError
        # xdata.sort()
        hist, x, y = np.histogram2d(xdata, data[colname], bins=bincount, density=True)
        print(hist.shape, x.shape, y.shape)
        xpos, ypos = np.meshgrid(x[:-1], y[:-1], indexing="ij")
        
        t = ax.contourf(xpos, ypos, hist)
        for x in xdata:
            if x <= xpos.max() and x >= xpos.min():
                ax.axvline(x, linewidth=0.5,color='black', linestyle='dashed', alpha=0.9)


        # t = ax.scatter(data.MAX_MAGNITUDE, data[colname])

        ax.set_xlabel(f'{units[paramname]}', fontsize=12)
        ax.set_ylabel(r'Estimated Accuracy, 1$\sigma$ [arcsec]', fontsize=12)
        # ax.set_title(f'{sensetitle[paramname]}\n', fontsize=15)
        
        if params[0] == 'FOCAL_LENGTH':
            fovcalc = lambda f: np.round(2*np.rad2deg(np.arctan(c.SENSOR_HEIGHT/(2*f))),2)
            tls = np.array([1000,2000,3000,4000,5000,6000])
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xlabel('FOV [deg]', fontsize=12)
            ax2.set_xticklabels(fovcalc(tls))
            # ax.set_title('Focal Length [px]')
            ax.set_xlabel('Focal Length [px]')
            plt.subplots_adjust(hspace=0.5)#, left=.104, right=.957)
        else:
            plt.subplots_adjust(top=0.962, hspace=0.5)#, left=.104, right=.957)
        
        
        
        
        cbar = fig.colorbar(t, ax=ax)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Probability Density', rotation=90, fontsize=12)
        
        ttil = '/Users/gagandeepthapar/Desktop/School/AERO/MS/Thesis/Documents/StarTrackerThesis/chapters/5_Univariate_Effect_Analysis/images/' +pname+'.png'
        if args.p:
            plt.show()
        else:
            plt.savefig(ttil, dpi=600)

        # plt.show()

    if args.s == 'c':
        cname = 'MAXIMUM MAGNITUDE VISIBLE'
        fig, ax = plt.subplots()

        halfnormal = lambda x, s: np.sqrt(2)/(s*np.sqrt(np.pi)) * np.exp(-x**2 / (2*s**2))

        x_data = halfnormal(np.linspace(0, 2, 100), datadf[cname].std() * ((1 - 2/np.pi)**(0.5)))[::-1]



        print(datadf[cname].min())
        print(datadf[cname].max())
        print(datadf[cname].std() * (np.sqrt(1-2/np.pi)))
        ax.hist((datadf[cname]), bins=50, density=True, label='Simulation Data')
        ax.plot(np.linspace(0,2, 100)+3.35, x_data, 'r', label='Half-Normal Distribution')
        # ax.set_title('Maximum Magnitude in given image'.title(), fontsize=15)
        ax.set_xlabel('Magnitude', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ttil = '/Users/gagandeepthapar/Desktop/School/AERO/MS/Thesis/Documents/StarTrackerThesis/chapters/5_Univariate_Effect_Analysis/images/' +'MAX_MAG'+'.png'
        # plt.savefig(ttil, dpi=600)
        if args.p:
            plt.show()
        else:
            plt.savefig(ttil, dpi=600)
        
        # ax.hist(datadf[colname])