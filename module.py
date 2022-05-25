import numpy as np

EFFECTIVITY = [1.0000000000,
               0.8691199806,
               0.5705831430,
               0.2829551534,
                0.1059926494
               ]

def multiply_w_penalty(mods):
    mods = np.sort(mods)[::-1]
    group_mod = 0
    for i, mod in enumerate(mods):
        #pen = EFFECTIVITY[i]**(i**2)
        pen = EFFECTIVITY[i]
        #pen = 0.87**(i**2)
        mod_pen = (mod * pen)

        #1 - (1 - Mod[1]*Pen[1]) * (1 - Mod[2]*Pen[2]) * (1 - Mod[3]*Pen[3]) * ... * (1 - Mod[n]*Pen[n])


        if i == 0:
            group_mod = mod_pen
        else:
            group_mod *= mod_pen/10
        print('mod:', mod, ', pen', pen, 'after pen:', mod_pen, 'group:', group_mod)

    return group_mod



def veloc(full, pers):
    return full - (pers*full/100)


#[08:40:25] Vedja Scrappy > full 3456
# [08:40:42] Vedja Scrappy > 90 429
#set_1 = [98, 96]
set_1 = [90, 90, 90]
out = multiply_w_penalty(set_1)
#out = penalty2(set_1)
print(out, '%')
print(veloc(full=3456, pers=out), 'm/s')

