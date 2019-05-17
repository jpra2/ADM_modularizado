from definitions.functions1 import lu_inv4


class OperatorsAms1:

    def __init__(self):
        op1 = Operators1(As, )




class Operators1:

    def __init__(self):
        self.tags = dict()

    def calc_OP1(self, As, wire_numbers1):

        na = wire_numbers1[2]
        ids_arestas=np.where(As['Aev'].sum(axis=1)==0)[0]
        ids_arestas_slin_m0=np.setdiff1d(range(na),ids_arestas)

        invAee=lu_inv4(As['Aee'],ids_arestas_slin_m0)
        M2=-invAee*As['Aev']
        PAD=vstack([M2,As['Ivv']]) #P=np.concatenate((-np.dot(np.linalg.inv(Aee),Aev),Ivv), axis=0)

        #invAff=lu_inv4(Aff,ids_faces_slin_m0)
        invAff=invbAff
        M3=-invAff*(As['Afe']*M2)

        #ids_internos_slin_m0=np.setdiff1d(ids_internos_slin_m0,IDs_internos_0_locais)
        PAD=vstack([M3,PAD])   #P=np.concatenate((-np.dot(np.linalg.inv(Aff),np.dot(Afe,P[0:na,0:nv])),P), axis=0)
        #invAii=lu_inv4(Aii,ids_internos_slin_m0)
        invAii=invbAii
        PAD=vstack([-invAii*(Aif*M3),PAD]) ##P=np.concatenate((np.dot(-np.linalg.inv(Aii),np.dot(Aif,P[0:nf,0:nv])),P),axis=0)
        print("get_OP_AMS", time.time()-ta1)

        del(M3)

        ids_1=M1.mb.tag_get_data(L1_ID_tag,vertices,flat=True)
        ids_class=M1.mb.tag_get_data(fine_to_primal1_classic_tag,vertices,flat=True)
        t0=time.time()

        AMS_TO_ADM=dict(zip(ids_class,ids_1))
        ty=time.time()
        vm=M1.mb.create_meshset()
        M1.mb.add_entities(vm,vertices)

        tm=time.time()
        PAD=csc_matrix(PAD)
        OP3=PAD.copy()
        nivel_0=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
        tor=time.time()
        ID_global1=M1.mb.tag_get_data(M1.ID_reordenado_tag,nivel_0, flat=True)
        IDs_ADM1=M1.mb.tag_get_data(L1_ID_tag,nivel_0, flat=True)
        IDs_AMS1=M1.mb.tag_get_data(fine_to_primal1_classic_tag,nivel_0, flat=True)
        OP3[ID_global1]=csc_matrix((1,OP3.shape[1]))
        IDs_ADM1=M1.mb.tag_get_data(L1_ID_tag,nivel_0, flat=True)
        IDs_ADM_1=M1.mb.tag_get_data(L1_ID_tag,vertices, flat=True)
        IDs_AMS_1=M1.mb.tag_get_data(fine_to_primal1_classic_tag,vertices, flat=True)
        lp=IDs_AMS_1
        cp=IDs_ADM_1
        dp=np.repeat(1,len(lp))
        permut=csc_matrix((dp,(lp,cp)),shape=(len(vertices),n1))
        opad3=OP3*permut
        m=find(opad3)
        l1=m[0]
        c1=m[1]
        d1=m[2]
        l1=np.concatenate([l1,ID_global1])
        c1=np.concatenate([c1,IDs_ADM1])
        d1=np.concatenate([d1,np.ones(len(nivel_0))])
        opad3=csc_matrix((d1,(l1,c1)),shape=(len(M1.all_volumes),n1))
        print("opad1",tor-time.time(),time.time()-ta1, time.time()-tempo0_ADM)
        OP_ADM=csc_matrix(opad3)

        print("obteve OP_ADM_1",time.time()-tempo0_ADM)

        l1=M1.mb.tag_get_data(L1_ID_tag, M1.all_volumes, flat=True)
        c1=M1.mb.tag_get_data(M1.ID_reordenado_tag, M1.all_volumes, flat=True)
        d1=np.ones((1,len(l1)),dtype=np.int)[0]
        OR_ADM=csc_matrix((d1,(l1,c1)),shape=(n1,len(M1.all_volumes)))

        l1=M1.mb.tag_get_data(fine_to_primal1_classic_tag, M1.all_volumes, flat=True)
        c1=M1.mb.tag_get_data(M1.ID_reordenado_tag, M1.all_volumes, flat=True)
        d1=np.ones((1,len(l1)),dtype=np.int)[0]
        OR_AMS=csc_matrix((d1,(l1,c1)),shape=(nv,len(M1.all_volumes)))

        OP_AMS=PAD
