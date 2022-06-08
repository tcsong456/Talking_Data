def find_middle(d):
    d.loc[d['device_model'].map(lambda x:x.startswith('802')),'device_model'] = d.loc[d['device_model'].map(lambda x:x.startswith('802')),'device_model'].map(lambda x:x[:-1]+' '+x[-1])
    d.loc[d['device_model']=='大Q Note','device_model'] = '大QNote'
    d.loc[d['device_model']=='EVO 3D X515m','device_model'] = 'EVO3DX515m'
    d.loc[d['device_model']=='M8St','device_model'] = 'M8 St'
    d.loc[d['device_model']=='M8x','device_model'] = 'M8 x'
    d.loc[d['device_model']=='Sensation XE with Beats Audio Z715e','device_model'] = 'Sensation XEwithBeatsAudioZ715e'
    m1 = d['phone_brand'].map(lambda x:x.startswith('HTC'))
    m2 = d['device_model'].map(lambda x:x.startswith('T32'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('LG'))
    m2 = d['device_model'].map(lambda x:x.startswith('F'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('LG'))
    m2 = d['device_model'].map(lambda x:x.startswith('G'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('LG'))
    m2 = d['device_model'].map(lambda x:x.startswith('L'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('LG'))
    m2 = d['device_model'].map(lambda x:x.startswith('P'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('OPPO'))
    m2 = d['device_model'].map(lambda x:x.startswith('A'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('TCL'))
    m2 = d['device_model'].map(lambda x:x.startswith('P3'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('TCL'))
    m2 = d['device_model'].map(lambda x:x.startswith('P5'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('TCL'))
    m2 = d['device_model'].map(lambda x:x.startswith('S'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('vivo'))
    m2 = d['device_model'].map(lambda x:x.startswith('E'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('vivo'))
    m2 = d['device_model'].map(lambda x:x.startswith('S'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    d.loc[d['device_model'].map(lambda x:x.startswith('Galaxy Tab')),'device_model'] = 'GalaxyTab'
    m1 = d['phone_brand'].map(lambda x:x.startswith('华为'))
    m2 = d['device_model'].map(lambda x:x.startswith('Y3'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('华为'))
    m2 = d['device_model'].map(lambda x:x.startswith('Y5'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('华为'))
    m2 = d['device_model'].map(lambda x:x.startswith('Y6'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('华为'))
    m2 = d['device_model'].map(lambda x:x.startswith('畅享'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['device_model'].map(lambda x:x.startswith('荣耀'))
    m2 = ~d['device_model'].map(lambda x:x.startswith('荣耀畅玩'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    d.loc[d['device_model'].map(lambda x:x.startswith('荣耀畅玩')),'device_model'] = d.loc[d['device_model'].map(lambda x:x.startswith('荣耀畅玩')),
          'device_model'].map(lambda x:x[:4]+' '+x[4:])
    d.loc[d['device_model'].map(lambda x:x.startswith('麦芒')),'device_model'] = d.loc[d['device_model'].map(lambda x:x.startswith('麦芒')),
          'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('天语'))
    m2 = d['device_model'].map(lambda x:x.startswith('L'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    d.loc[d['device_model'].map(lambda x:x.startswith('T60'or'T619'or'T619+'or'T621'or'T780+'or'T85+'or'T87+'or'T91')),'device_model'] = d.loc[d['device_model'].map(lambda x:x.startswith('T60'or'T619'or'T619+'or'T621'or'T780+'or'T85+'or'T87+'or'T91')),
          'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('富可视'))
    m2 = d['device_model'].map(lambda x:x.startswith('M'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    d.loc[d['device_model'].map(lambda x:x.startswith('红米')),'device_model'] = d.loc[d['device_model'].map(lambda x:x.startswith('红米')),
          'device_model'].map(lambda x:x[:2]+' '+x[2:])
    d.loc[d['device_model'].map(lambda x:x.startswith('红米火星一号探索版')),'device_model'] = d.loc[d['device_model'].map(lambda x:x.startswith('红米火星一号探索版')),'device_model'].map(lambda x:x[:4]+' '+x[4:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('摩托罗拉'))
    m2 = d['device_model'].map(lambda x:x.startswith('XT'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('朵唯'))
    m2 = d['device_model'].map(lambda x:x.startswith('D'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('朵唯'))
    m2 = d['device_model'].map(lambda x:x.startswith('L'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('朵唯'))
    m2 = d['device_model'].map(lambda x:x.startswith('S'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('朵唯'))
    m2 = d['device_model'].map(lambda x:x.startswith('T'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('朵唯'))
    m2 = d['device_model'].map(lambda x:x.startswith('倾城'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('海信'))
    m2 = d['device_model'].map(lambda x:x.startswith('e'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('海信'))
    m2 = d['device_model'].map(lambda x:x.startswith('U'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('海信'))
    m2 = d['device_model'].map(lambda x:x.startswith('X'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('联想'))
    m2 = d['device_model'].map(lambda x:x.startswith('A2'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('联想'))
    m2 = d['device_model'].map(lambda x:x.startswith('A3'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('联想'))
    m2 = d['device_model'].map(lambda x:x.startswith('A5'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('联想'))
    m2 = d['device_model'].map(lambda x:x.startswith('A6'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('联想'))
    m2 = d['device_model'].map(lambda x:x.startswith('A7'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('联想'))
    m2 = d['device_model'].map(lambda x:x.startswith('A8'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('联想'))
    m2 = d['device_model'].map(lambda x:x.startswith('K'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('联想'))
    m2 = d['device_model'].map(lambda x:x.startswith('P'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('联想'))
    m2 = d['device_model'].map(lambda x:x.startswith('S6'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('联想'))
    m2 = d['device_model'].map(lambda x:x.startswith('S7'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('联想'))
    m2 = d['device_model'].map(lambda x:x.startswith('S8'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('联想'))
    m2 = d['device_model'].map(lambda x:x.startswith('S9'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['device_model'].map(lambda x:'黄金斗士' in x)
    d.loc[m1,'device_model'] = '黄金斗士'
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷比'))
    m2 = d['device_model'].map(lambda x:x.startswith('S'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('52'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('53'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('58'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('59'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('72'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('80'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('81'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('80'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('82'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('87'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('91'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('99'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('Y'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('大神'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('锋尚'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('大神'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('酷派'))
    m2 = d['device_model'].map(lambda x:x.startswith('大神'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    d.loc[d['device_model'].map(lambda x:x.startswith('E3'or'E5'or'E6'or'E6T'or'E7'or'E8')),'device_model'] = d.loc[d['device_model'].map(lambda x:x.startswith('E3'or'E5'or'E6'or'E6T'or'E7'or'E8')),
          'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('金立'))
    m2 = d['device_model'].map(lambda x:x.startswith('F'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('金立'))
    m2 = d['device_model'].map(lambda x:x.startswith('GN'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('金立'))
    m2 = d['device_model'].map(lambda x:x.startswith('M'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('金立'))
    m2 = d['device_model'].map(lambda x:x.startswith('S'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('金立'))
    m2 = d['device_model'].map(lambda x:x.startswith('V'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('金立'))
    m2 = d['device_model'].map(lambda x:x.startswith('X'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:1]+' '+x[1:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('金立'))
    m2 = d['device_model'].map(lambda x:x.startswith('天鉴'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('魅族'))
    m2 = d['device_model'].map(lambda x:x.startswith('M0'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('魅族'))
    m2 = d['device_model'].map(lambda x:x.startswith('MX'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    m1 = d['phone_brand'].map(lambda x:x.startswith('魅族'))
    m2 = d['device_model'].map(lambda x:x.startswith('魅蓝'))
    d.loc[m1&m2,'device_model'] = d.loc[m1&m2,'device_model'].map(lambda x:x[:2]+' '+x[2:])
    
    d.loc[d['device_model']=='G Pro Lite Dual','device_model'] = 'G ProLiteDual'
    d.loc[d['device_model']=='Optimus G Pro','device_model'] = 'Optimus GPro'
    d.loc[d['device_model']=='Optimus G Pro 2','device_model'] = 'Optimus GPro2'
    d.loc[d['device_model']=='N1T','device_model'] = 'N1 T'
    d.loc[d['device_model']=='N1W','device_model'] = 'N1 W'
    d.loc[d['device_model']=='R1C','device_model'] = 'R1 C'
    d.loc[d['device_model']=='R1S','device_model'] = 'R1 S'
    d.loc[d['device_model']=='R2010','device_model'] = 'R20'
    d.loc[d['device_model']=='R2017','device_model'] = 'R20'
    d.loc[d['device_model']=='R3','device_model'] = 'R 3'
    d.loc[d['device_model']=='R5','device_model'] = 'R 5'
#    d.loc[d['device_model']=='R7'] = 'R 7'
    d.loc[d['device_model']=='R7 Plus','device_model'] = 'R7 Plus'
    d.loc[d['device_model']=='R7s','device_model'] = 'R7 s'
    d.loc[d['device_model']=='R801','device_model'] = 'R80 1'
    d.loc[d['device_model']=='R805','device_model'] = 'R80 5'
    d.loc[d['device_model']=='R807','device_model'] = 'R80 7'
    d.loc[d['device_model']=='R809T','device_model'] = 'R80 9T'
    d.loc[d['device_model']=='R811','device_model'] = 'R81 1'
    d.loc[d['device_model']=='R813T','device_model'] = 'R81 3T'
    d.loc[d['device_model']=='R815T','device_model'] = 'R81 5T'
    d.loc[d['device_model']=='R817','device_model'] = 'R81 7'
    d.loc[d['device_model']=='R817T','device_model'] = 'R81 7T'
    d.loc[d['device_model']=='R819T','device_model'] = 'R81 9T'
    d.loc[d['device_model']=='R820','device_model'] = 'R82 0'
    d.loc[d['device_model']=='R8205','device_model'] = 'R82 05'
    d.loc[d['device_model']=='R821T','device_model'] = 'R82 1T'
    d.loc[d['device_model']=='R823T','device_model'] = 'R82 3T'
    d.loc[d['device_model']=='R827T','device_model'] = 'R82 7T'
    d.loc[d['device_model']=='R830','device_model'] = 'R83 0'
    d.loc[d['device_model']=='R830S','device_model'] = 'R83 0S'
    d.loc[d['device_model']=='R831S','device_model'] = 'R83 1S'
    d.loc[d['device_model']=='R831T','device_model'] = 'R83 1T'
    d.loc[d['device_model']=='R833T','device_model'] = 'R83 3T'
    d.loc[d['device_model']=='R9 Plus','device_model'] = 'R9 Plus'
    d.loc[d['device_model']=='U701','device_model'] = 'U 701'
    d.loc[d['device_model']=='U701T','device_model'] = 'U 701T'
    d.loc[d['device_model']=='U705T','device_model'] = 'U 705T'
    d.loc[d['device_model']=='U707T','device_model'] = 'U 707T'
    d.loc[d['device_model']=='么么哒3N','device_model'] = 'U么么哒 3N'
    d.loc[d['device_model']=='么么哒3S','device_model'] = 'U么么哒 3S'
    d.loc[d['device_model']=='X1S','device_model'] = 'X1 S'
    d.loc[d['device_model']=='X1ST','device_model'] = 'X1 ST'
    d.loc[d['device_model']=='X3F','device_model'] = 'X3 F'
    d.loc[d['device_model']=='X3L','device_model'] = 'X3 L'
    d.loc[d['device_model']=='X3S','device_model'] = 'X3 S'
    d.loc[d['device_model']=='X3T','device_model'] = 'X3 T'
    d.loc[d['device_model']=='X3V','device_model'] = 'X3 V'
    d.loc[d['device_model']=='X5L','device_model'] = 'X5 L'
    d.loc[d['device_model']=='X5M','device_model'] = 'X5 M'
    d.loc[d['device_model']=='X5Max','device_model'] = 'X5 Max'
    d.loc[d['device_model']=='X5Max+','device_model'] = 'X5 Max+'
    d.loc[d['device_model']=='X5Pro','device_model'] = 'X5 Pro'
    d.loc[d['device_model']=='X5SL','device_model'] = 'X5 SL'
    d.loc[d['device_model']=='X5V','device_model'] = 'X5 V'
    d.loc[d['device_model']=='X5Pro','device_model'] = 'X5 Pro'
    d.loc[d['device_model']=='X6 Plus D','device_model'] = 'X6 PlusD'
    d.loc[d['device_model']=='X710L','device_model'] = 'X7 10L'
    d.loc[d['device_model']=='Xplay3S','device_model'] = 'Xplay 3S'
    d.loc[d['device_model']=='Y11IT','device_model'] = 'Y11 IT'
    d.loc[d['device_model']=='Y11IW','device_model'] = 'Y11 IW'
    d.loc[d['device_model']=='Y13L','device_model'] = 'Y13 L'
    d.loc[d['device_model']=='Y13T','device_model'] = 'Y13 T'
    d.loc[d['device_model']=='Y13iL','device_model'] = 'Y13 iL'
    d.loc[d['device_model']=='Y17T','device_model'] = 'Y17 T'
    d.loc[d['device_model']=='Y17W','device_model'] = 'Y17 W'
    d.loc[d['device_model']=='Y18L','device_model'] = 'Y18 L'
    d.loc[d['device_model']=='Y19T','device_model'] = 'Y1920'
    d.loc[d['device_model']=='Y20T','device_model'] = 'Y1920'
    d.loc[d['device_model']=='Y22IL','device_model'] = 'Y22 IL'
    d.loc[d['device_model']=='Y22L','device_model'] = 'Y22 L'
    d.loc[d['device_model']=='Y23L','device_model'] = 'Y23 L'
    d.loc[d['device_model']=='Y31','device_model'] = 'Y3 1'
    d.loc[d['device_model']=='Y33','device_model'] = 'Y3 3'
    d.loc[d['device_model']=='Y35','device_model'] = 'Y3 5'
    d.loc[d['device_model']=='Y37','device_model'] = 'Y3 7'
    d.loc[d['device_model']=='Y3T','device_model'] = 'Y3 T'
    d.loc[d['device_model']=='Y613F','device_model'] = 'Y613 F'
    d.loc[d['device_model']=='Y622','device_model'] = 'Y62 2'
    d.loc[d['device_model']=='Y622','device_model'] = 'Y62 2'
    d.loc[d['device_model']=='Y623','device_model'] = 'Y62 3'
    d.loc[d['device_model']=='Y627','device_model'] = 'Y62 7'
    d.loc[d['device_model']=='Y628','device_model'] = 'Y62 8'
    d.loc[d['device_model']=='Y913','device_model'] = 'Y9 13'
    d.loc[d['device_model']=='Y923','device_model'] = 'Y9 23'
    d.loc[d['device_model']=='Y927','device_model'] = 'Y9 27'
    d.loc[d['device_model']=='Y928','device_model'] = 'Y9 28'
    d.loc[d['device_model']=='Y937','device_model'] = 'Y9 37'
    d.loc[d['device_model']=='2016版 Galaxy A5','device_model'] = '2016版Galaxy A5'
    d.loc[d['device_model']=='2016版 Galaxy A7','device_model'] = '2016版Galaxy A7'
    d.loc[d['device_model']=='2016版 Galaxy A9','device_model'] = '2016版Galaxy A9'
    d.loc[d['device_model']=='G3588V','device_model'] = 'G3 588V'
    d.loc[d['device_model']=='G3609','device_model'] = 'G3 609'
    d.loc[d['device_model']=='G3818','device_model'] = 'G3 818'
    d.loc[d['device_model']=='G3819D','device_model'] = 'G3 819D'
    d.loc[d['device_model']=='Galaxy Ace','device_model'] = 'GalaxyAce'
    d.loc[d['device_model']=='Galaxy Ace 2','device_model'] = 'GalaxyAce 2'
    d.loc[d['device_model']=='Galaxy Ace 3','device_model'] = 'GalaxyAce 3'
    d.loc[d['device_model']=='Galaxy Ace 4','device_model'] = 'GalaxyAce 4'
    d.loc[d['device_model']=='Galaxy Ace DUOS','device_model'] = 'GalaxyAce DUOS'
    d.loc[d['device_model']=='Galaxy Ace Plus','device_model'] = 'GalaxyAce Plus'
    d.loc[d['device_model']=='Galaxy Alpha','device_model'] = 'GalaxyAlpha'
    d.loc[d['device_model']=='Galaxy E7','device_model'] = 'GalaxyE7'
    d.loc[d['device_model']=='Galaxy Fame','device_model'] = 'GalaxyFame'
    d.loc[d['device_model']=='Galaxy Gio','device_model'] = 'GalaxyGio'
    d.loc[d['device_model']=='Galaxy Golden','device_model'] = 'GalaxyGolden'
    d.loc[d['device_model']=='Galaxy Core','device_model'] = 'GalaxyCore'
    d.loc[d['device_model']=='Galaxy Core 2','device_model'] = 'GalaxyCore 2'
    d.loc[d['device_model']=='Galaxy Core 4G','device_model'] = 'GalaxyCore 4G'
    d.loc[d['device_model']=='Galaxy Core Advance','device_model'] = 'GalaxyCore Advance'
    d.loc[d['device_model']=='Galaxy Core Lite','device_model'] = 'GalaxyCore Lite'
    d.loc[d['device_model']=='Galaxy Core Max','device_model'] = 'GalaxyCore Max'
    d.loc[d['device_model']=='Galaxy Core Mini','device_model'] = 'GalaxyCore Mini'
    d.loc[d['device_model']=='Galaxy Core Prime','device_model'] = 'GalaxyCore Prime'
    d.loc[d['device_model']=='Galaxy Grand','device_model'] = 'GalaxyGrand 2'
    d.loc[d['device_model']=='Galaxy Grand 2 LTE','device_model'] = 'GalaxyGrand 2LTE'
    d.loc[d['device_model']=='Galaxy Grand DUOS','device_model'] = 'GalaxyGrand DUOS'
    d.loc[d['device_model']=='Galaxy Grand Neo Plus','device_model'] = 'GalaxyGrand NeoPlus'
    d.loc[d['device_model']=='Galaxy Grand Prime','device_model'] = 'GalaxyGrand Prime'
    d.loc[d['device_model']=='Galaxy J','device_model'] = 'GalaxyJ'
    d.loc[d['device_model']=='Galaxy J1','device_model'] = 'GalaxyJ 1'
    d.loc[d['device_model']=='Galaxy J3','device_model'] = 'GalaxyJ 3'
    d.loc[d['device_model']=='Galaxy J5','device_model'] = 'GalaxyJ 5'
    d.loc[d['device_model']=='Galaxy J7','device_model'] = 'GalaxyJ 7'
    d.loc[d['device_model']=='Galaxy K Zoom','device_model'] = 'GGalaxyK Zoom'
    d.loc[d['device_model']=='Galaxy Nexus','device_model'] = 'GalaxyNexus'
    d.loc[d['device_model']=='Galaxy On7','device_model'] = 'GalaxyOn 7'
    d.loc[d['device_model']=='Galaxy On5','device_model'] = 'GalaxyOn 5'
    d.loc[d['device_model']=='Galaxy Premier','device_model'] = 'GalaxyPremier'
    d.loc[d['device_model']=='Galaxy R','device_model'] = 'GalaxyR'
    d.loc[d['device_model']=='Galaxy S2 Plus','device_model'] = 'GalaxyS2 Plus'
    d.loc[d['device_model']=='Galaxy Y','device_model'] = 'GalaxyY'
    d.loc[d['device_model']=='Galaxy K Zoom','device_model'] = 'GalaxyK Zoom'
    d.loc[d['device_model']=='Galaxy Mega 2','device_model'] = 'GalaxyMega 2'
    d.loc[d['device_model']=='Galaxy Mega 5.8','device_model'] = 'GalaxMega 5.8'
    d.loc[d['device_model']=='Galaxy Mega 6.3','device_model'] = 'GalaxyMega 6.3'
    d.loc[d['device_model']=='Galaxy Mega Plus','device_model'] = 'GalaxyMega Plus'
    d.loc[d['device_model']=='Galaxy Note','device_model'] = 'GalaxyNote'
    d.loc[d['device_model']=='Galaxy Note 10.1','device_model'] = 'GalaxyNote 10.1'
    d.loc[d['device_model']=='Galaxy Note 10.1 2014 Edition P601','device_model'] = 'GalaxyNote 10.1 2014 Edition P601'
    d.loc[d['device_model']=='Galaxy Note 2','device_model'] = 'GalaxyNote 2'
    d.loc[d['device_model']=='Galaxy Note 3','device_model'] = 'GalaxyNote 3'
    d.loc[d['device_model']=='Galaxy Note 3 Lite','device_model'] = 'GalaxyNote 3Lite'
    d.loc[d['device_model']=='Galaxy Note 4','device_model'] = 'GalaxyNote 4'
    d.loc[d['device_model']=='Galaxy Note 5','device_model'] = 'GalaxyNote 5'
    d.loc[d['device_model']=='Galaxy Note 8.0','device_model'] = 'GalaxyNote 8.0'
    d.loc[d['device_model']=='Galaxy Note Edge','device_model'] = 'GalaxyNote Edge'
    d.loc[d['device_model']=='Galaxy S','device_model'] = 'GalaxyS'
    d.loc[d['device_model']=='Galaxy S Advance','device_model'] = 'GalaxyS Advance'
    d.loc[d['device_model']=='Galaxy S DUOS','device_model'] = 'GalaxyS DUOS'
    d.loc[d['device_model']=='Galaxy S DUOS 2','device_model'] = 'GalaxyS DUOS2'
    d.loc[d['device_model']=='Galaxy S L','device_model'] = 'GalaxyS L'
    d.loc[d['device_model']=='Galaxy S Plus','device_model'] = 'GalaxyS Plus'
    d.loc[d['device_model']=='Galaxy S2','device_model'] = 'GalaxyS2'
    d.loc[d['device_model']=='Galaxy S2 HD LTE E120S','device_model'] = 'GalaxyS2 HD LTE E120S'
    d.loc[d['device_model']=='Galaxy S3','device_model'] = 'GalaxyS3'
    d.loc[d['device_model']=='Galaxy S3 Mini','device_model'] = 'GalaxyS3 Mini'
    d.loc[d['device_model']=='Galaxy S3 Neo+','device_model'] = 'GalaxyS3 Neo+'
    d.loc[d['device_model']=='Galaxy S4','device_model'] = 'GalaxyS4'
    d.loc[d['device_model']=='Galaxy S4 Active','device_model'] = 'GalaxyS4 Active'
    d.loc[d['device_model']=='Galaxy S4 Mini','device_model'] = 'GalaxyS4 Mini'
    d.loc[d['device_model']=='Galaxy S4 Zoom','device_model'] = 'GalaxyS4 Zoom'
    d.loc[d['device_model']=='Galaxy S5','device_model'] = 'Galaxy S5'
    d.loc[d['device_model']=='Galaxy S5 Plus','device_model'] = 'GalaxyS5 Plus'
    d.loc[d['device_model']=='Galaxy S6','device_model'] = 'GalaxyS6'
    d.loc[d['device_model']=='Galaxy S6 Edge','device_model'] = 'GalaxyS6 Edge'
    d.loc[d['device_model']=='Galaxy S6 Edge+','device_model'] = 'GalaxyS6 Edge+'
    d.loc[d['device_model']=='Galaxy S7','device_model'] = 'GalaxyS7'
    d.loc[d['device_model']=='Galaxy S7 Edge','device_model'] = 'GalaxyS7 Edge'
    d.loc[d['device_model']=='Galaxy Style DUOS','device_model'] = 'GalaxyStyle DUOS'
    d.loc[d['device_model']=='Galaxy W','device_model'] = 'GalaxyW'
    d.loc[d['device_model']=='Galaxy Y','device_model'] = 'GalaxyY'
    d.loc[d['device_model'].map(lambda x:x.startswith('Galaxy Trend')),'device_model'] = 'GalaxyTrend'
    d.loc[d['device_model'].map(lambda x:x.startswith('Galaxy Win')),'device_model'] = 'GalaxyWin'
    d.loc[d['device_model']=='I8250','device_model'] = 'I82 50'
    d.loc[d['device_model']=='I8258','device_model'] = 'I82 58'
    d.loc[d['device_model']=='I8268','device_model'] = 'I82 68'
    d.loc[d['device_model']=='I9050','device_model'] = 'I9 050'
    d.loc[d['device_model']=='I9118','device_model'] = 'I9 118'
    d.loc[d['device_model']=='S5300','device_model'] = 'S5 300'
    d.loc[d['device_model']=='S5830I','device_model'] = 'S5 830I'
    d.loc[d['device_model']=='S7566','device_model'] = 'S7 566'
    d.loc[d['device_model']=='S7568','device_model'] = 'S7 568'
    d.loc[d['device_model']=='S7568I','device_model'] = 'S7 568I'
    d.loc[d['device_model']=='S7898','device_model'] = 'S7 898'
    d.loc[d['device_model']=='W2013','device_model'] = 'W 2013'
    d.loc[d['device_model']=='W2014','device_model'] = 'W 2014'
    d.loc[d['device_model']=='W2015','device_model'] = 'W 2015'
    d.loc[d['device_model']=='W999','device_model'] = 'W 999'
    d.loc[d['device_model']=='G717C','device_model'] = 'G 717C'
    d.loc[d['device_model']=='G718C','device_model'] = 'G 718C'
    d.loc[d['device_model']=='N798','device_model'] = 'N 798'
    d.loc[d['device_model']=='N798+','device_model'] = 'N 798+'
    d.loc[d['device_model']=='N818','device_model'] = 'N 818'
    d.loc[d['device_model']=='N880E','device_model'] = 'N 880E'
    d.loc[d['device_model']=='N880F','device_model'] = 'N 880F'
    d.loc[d['device_model']=='N881F','device_model'] = 'N 881F'
    d.loc[d['device_model']=='N900','device_model'] = 'N 900'
    d.loc[d['device_model']=='N909','device_model'] = 'N 909'
    d.loc[d['device_model']=='N919','device_model'] = 'N 919'
    d.loc[d['device_model']=='N919D','device_model'] = 'N 919D'
    d.loc[d['device_model']=='N986','device_model'] = 'N 986'
    d.loc[d['device_model']=='Q201T','device_model'] = 'Q 201T'
    d.loc[d['device_model']=='Q301C','device_model'] = 'Q 301C'
    d.loc[d['device_model']=='Q302C','device_model'] = 'Q 302C'
    d.loc[d['device_model']=='Q501T','device_model'] = 'Q5 01T'
    d.loc[d['device_model']=='Q503U','device_model'] = 'Q5 03U'
    d.loc[d['device_model']=='Q505T','device_model'] = 'Q5 05T'
    d.loc[d['device_model']=='Q507T','device_model'] = 'Q5 07T'
    d.loc[d['device_model']=='Q519T','device_model'] = 'Q5 19T'
    d.loc[d['device_model']=='Q7','device_model'] = 'Q7'
    d.loc[d['device_model']=='Q701C','device_model'] = 'Q7 01C'
    d.loc[d['device_model']=='Q705U','device_model'] = 'Q7 05U'
    d.loc[d['device_model']=='Q801L','device_model'] = 'Q8 01L'
    d.loc[d['device_model']=='Q802T','device_model'] = 'Q8 02T'
    d.loc[d['device_model']=='Q805T','device_model'] = 'Q8 05T'
    d.loc[d['device_model'].map(lambda x:x.startswith('U7')),'device_model'] = d.loc[d['device_model'].map(lambda x:x.startswith('U7')),
          'device_model'].map(lambda x:x[:2]+' '+x[2:])
    d.loc[d['device_model'].map(lambda x:x.startswith('U8')),'device_model'] = d.loc[d['device_model'].map(lambda x:x.startswith('U8')),
          'device_model'].map(lambda x:x[:2]+' '+x[2:])
    d.loc[d['device_model'].map(lambda x:x.startswith('U9')),'device_model'] = d.loc[d['device_model'].map(lambda x:x.startswith('U9')),
          'device_model'].map(lambda x:x[:2]+' '+x[2:])
    d.loc[d['device_model'].map(lambda x:x.startswith('V8')),'device_model'] = d.loc[d['device_model'].map(lambda x:x.startswith('V8')),
          'device_model'].map(lambda x:x[:2]+' '+x[2:])
    d.loc[d['device_model'].map(lambda x:x.startswith('V9')),'device_model'] = d.loc[d['device_model'].map(lambda x:x.startswith('V9')),
          'device_model'].map(lambda x:x[:2]+' '+x[2:])

    d.loc[d['device_model']=='V5S','device_model'] = 'V5 S'
    d.loc[d['device_model']=='威武3','device_model'] = '威武 3'
    d.loc[d['device_model']=='威武3C','device_model'] = '威武 3C'
    d.loc[d['device_model']=='M601','device_model'] = 'M 601'
    d.loc[d['device_model']=='M701','device_model'] = 'M 701'
    d.loc[d['device_model']=='M811','device_model'] = 'M 811'
    d.loc[d['device_model']=='M812C','device_model'] = 'M 812C'
    d.loc[d['device_model']=='超级手机1s','device_model'] = '超级手机1 s'
    d.loc[d['device_model']=='UIMI3','device_model'] = 'UIMI 3'
    d.loc[d['device_model']=='UIMI4','device_model'] = 'UIMI 4'
    d.loc[d['device_model']=='Z5S','device_model'] = 'Z5 S'
    d.loc[d['device_model'].map(lambda x:x.startswith('C88')),'device_model'] = d.loc[d['device_model'].map(lambda x:x.startswith('C88')),
          'device_model'].map(lambda x:x[:2]+' '+x[2:])
    d.loc[d['device_model']=='G521','device_model'] = 'G52 1'
    d.loc[d['device_model']=='G525','device_model'] = 'G52 5'
    d.loc[d['device_model']=='G606','device_model'] = 'G6 06'
    d.loc[d['device_model']=='G610C','device_model'] = 'G6 10C'
    d.loc[d['device_model']=='G610S','device_model'] = 'G6 10S'
    d.loc[d['device_model']=='G610T','device_model'] = 'G6 10T'
    d.loc[d['device_model']=='G615 U10','device_model'] = 'G6 15 U10'
    d.loc[d['device_model']=='G616 L076','device_model'] = 'G6 16 L076'
    d.loc[d['device_model']=='G628','device_model'] = 'G600'
    d.loc[d['device_model']=='G629','device_model'] = 'G600'
    d.loc[d['device_model']=='G630','device_model'] = 'G600'
    d.loc[d['device_model']=='G716','device_model'] = 'G 716'
    d.loc[d['device_model']=='M811','device_model'] = 'M 811'
    d.loc[d['device_model']=='M812C','device_model'] = 'M 812C'
    d.loc[d['device_model']=='T8620','device_model'] = 'T 8620'
    d.loc[d['device_model']=='T8830Pro','device_model'] = 'T 8830Pro'
    d.loc[d['device_model']=='U9508','device_model'] = 'U 9508'

    return d