
import os
import misctools

def wget(s,fname,Dir):
    os.system('wget '+s+fname+' -O ../data/inp/'+Dir+'/'+fname)
    #os.system('mv '+fname+' '+Dir+'/')

root = 'http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/'
nres = 13

density = False
angle = False

'''
zs = 66 #cmb
zs = 16 #1.0
zs = 9  #0.5
zs = 21 #1.5
#zs = 25 #2.0
'''

for zs in [16]:
    # convergence map
    for i in range(0,108): 
        fname = 'allskymap_nres'+str(nres)+'r'+str(i).zfill(3)+'.zs'+str(zs)+'.mag.dat'
        if misctools.check_path('../data/inp/kappa/'+fname): continue
        wget(root+'sub'+str(int(i/54)+1)+'/nres'+str(nres)+'/',fname,'kappa')

# density map
if density:
    for i in range(1,108): 
        fname = 'allskymap_nres'+str(nres)+'r'+str(i).zfill(3)+'.delta_shell.dat'
        if misctools.check_path('delta/'+fname): continue
        if i<=53:  wget(root+'sub1/nres'+str(nres)+'/delta_shell_maps/',fname,'delta')
        if i>=54:  wget(root+'sub2/nres'+str(nres)+'/delta_shell_maps/',fname,'delta')

# deflection angle
if angle:
    for i in range(1,108): 
        fname = 'allskymap_nres'+str(nres)+'r'+str(i).zfill(3)+'.zs66.src.dat'
        if misctools.check_path('angle/'+fname): continue
        if i<=53:  wget(root+'sub1/nres'+str(nres)+'/',fname,'delta')
        if i>=54:  wget(root+'sub2/nres'+str(nres)+'/',fname,'delta')


