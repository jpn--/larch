import pandas
import os
import gzip
import pickle
import base64
import io
import numpy
from pytest import approx, raises

from ..data_warehouse import example_file

from .. import DataFrames

def test_dfs_info():

	from ..data_warehouse import example_file
	df = pandas.read_csv(example_file("MTCwork.csv.gz"))
	df.set_index(['casenum', 'altnum'], inplace=True)

	ds = DataFrames(df)

	s = io.StringIO()

	ds.info(out=s)

	assert s.getvalue() == (
		'larch.DataFrames:  (not computation-ready)\n'
		'  n_cases: 5029\n'
		'  n_alts: 6\n'
		'  data_ce: 36 variables, 22033 rows\n'
		'  data_co: <not populated>\n'
		'  data_av: <populated>\n')

	s = io.StringIO()
	ds.info(out=s, verbose=True)

	assert s.getvalue() == (
		'larch.DataFrames:  (not computation-ready)\n  n_cases: 5029\n  n_alts: 6\n  data_ce: 22033 rows\n'
		'    - chose    (22033 non-null int64)\n    - ivtt     (22033 non-null float64)\n'
		'    - ovtt     (22033 non-null float64)\n    - tottime  (22033 non-null float64)\n'
		'    - totcost  (22033 non-null float64)\n    - hhid     (22033 non-null int64)\n'
		'    - perid    (22033 non-null int64)\n    - numalts  (22033 non-null int64)\n'
		'    - dist     (22033 non-null float64)\n    - wkzone   (22033 non-null int64)\n'
		'    - hmzone   (22033 non-null int64)\n    - rspopden (22033 non-null float64)\n'
		'    - rsempden (22033 non-null float64)\n    - wkpopden (22033 non-null float64)\n'
		'    - wkempden (22033 non-null float64)\n    - vehavdum (22033 non-null int64)\n'
		'    - femdum   (22033 non-null int64)\n    - age      (22033 non-null int64)\n'
		'    - drlicdum (22033 non-null int64)\n    - noncadum (22033 non-null int64)\n'
		'    - numveh   (22033 non-null int64)\n    - hhsize   (22033 non-null int64)\n'
		'    - hhinc    (22033 non-null float64)\n    - famtype  (22033 non-null int64)\n'
		'    - hhowndum (22033 non-null int64)\n    - numemphh (22033 non-null int64)\n'
		'    - numadlt  (22033 non-null int64)\n    - nmlt5    (22033 non-null int64)\n'
		'    - nm5to11  (22033 non-null int64)\n    - nm12to16 (22033 non-null int64)\n'
		'    - wkccbd   (22033 non-null int64)\n    - wknccbd  (22033 non-null int64)\n'
		'    - corredis (22033 non-null int64)\n    - vehbywrk (22033 non-null float64)\n'
		'    - vocc     (22033 non-null int64)\n    - wgt      (22033 non-null int64)\n'
		'  data_co: <not populated>\n  data_av: <populated>\n')

	assert not ds.computational
	assert not ds.is_computational_ready()
	ds.computational = True
	assert ds.is_computational_ready()
	assert ds.computational
	s = io.StringIO()

	ds.info(out=s)

	assert s.getvalue() == (
		'larch.DataFrames:\n'
		'  n_cases: 5029\n'
		'  n_alts: 6\n'
		'  data_ce: 36 variables, 22033 rows\n'
		'  data_co: <not populated>\n'
		'  data_av: <populated>\n')

def test_service_idco():

	df = pandas.read_csv(example_file("MTCwork.csv.gz"))
	df.set_index(['casenum', 'altnum'], inplace=True)

	dfs = DataFrames(df, crack=True)
	check1 = dfs.make_idco('1')
	assert (check1 == 1).shape == (5029, 1)
	assert numpy.all(check1 == 1)

	check2 = dfs.make_idco('age')
	assert check2.shape == (5029, 1)
	assert numpy.all(check2.iloc[:5,0] == [35, 40, 28, 34, 43])
	assert numpy.all(check2.iloc[-5:,0] == [58, 33, 34, 35, 37])

	check3 = dfs.make_idco('age', '1')
	assert check3.shape == (5029, 2)



def test_dbf_reader():

	from .. import DBF
	from .. import data_warehouse

	q = DBF(data_warehouse.example_file('US-STATES.dbf'))

	assert q.fieldnames() == [
		'STATEFP',
		'STATENS',
		'AFFGEOID',
		'GEOID',
		'STUSPS',
		'NAME',
		'LSAD',
		'ALAND',
		'AWATER',
	]

	df = q.load_dataframe(preserve_order=False, strip_whitespace=False)

	correct = b'ABzY8<${%50{^v}dwdkt^}s_!9zhTV6cBJlk*FB=HM6sbNXP@i=EWu?0gUNnGs!O5?B<-EC4?Fg6%a&4#cB}W7GJGeA4P=<ilV4h#' \
			  b'EMT8pA{dKpS8czTHD?`bMMW~w7-9T?d0Q!^F8O@x#!+9=iWIJCzl)*>(o;%dZr+jMx&q=^$sJiSU2@jiy-HWl`x<Oa`PLG&dJGXY9' \
			  b'TGq|B^(?G%%$nGX))bN^dhTlQ%n=j<r|VS{@7}LmZvo2!rw+*R>eSj7&kZo-y*8Qbt!(8MWlW;QzZD>oKGUhUPa6BongclNU>6J37' \
			  b'1V{vFw*na~mS?z|>6V(C=I#B9vWgJE{BJTkP^3dclpR>AP10nIRC4ICSVk@=AUk)f;LIM%Vn?CvBHIKCm5II7ZMR3s-dU=56p!kGN' \
			  b'XQ4Ir#^&&W-va)jCKgf^uPvm?PPHbM=2xIMd6_yuP7Q?uPL6qwYYfGXqzF{y?YHEp6S`8=lX_YlFp<xK^SXw%#q^hhKCR$_IJjoiP' \
			  b'rglM1bq$<st-Z3aq69n^me&*(!zqRdlbbC)MeZG(TA5u9r$u2({NG2EtkS?lj0DwGgDkSA_rTQrcwT&1+=@<qoP_3#D4a<{qpgCo;' \
			  b'&wC=8cegJads36cr<3fIsK!NV}iGN?d+Yjk^OMYW%4K{2QqmylY^KX%;XRzhccPR<S-_WVRAT=BbYpv$&pMR$K>%$j$(2&lVg}Xfy' \
			  b'on@9LwZ5CdV^*5|a~{oXF%PCQoM4!{jMUPG<5{CQoB>3X`WZIhDzLCeL8<OeW7_avGCoGg-joIZS$)^fBpYGQebzNrg$3$q<v^Gr`' \
			  b'{+w{j{Fg`m+0N<L2$m8MB33K~t5+=#Xs!tsK&vo9exw`<Ea>u4<Qs9idNCP~lpy(@}|zBxZvr?HvtTlZZfo$vNLSIngMv%B{znj1?' \
			  b';@1I6<WmdM+y5(2Q3T>k?>zc9bLz*+m8~*qvT}xHGdpV71)8ZYQ=)Jr4*aHtz{jj#Yg0}aTef=)Ysjkag2htcExOGQ_-mCg&-hZy1' \
			  b'$RD+Zf!RcUk-02QbI=&_{gg5y?|g6gW*Y0Qvu|jmvK}Y>K=1j~>%a2Q+*$N(=~f!+a|3_gK>b`3JaRSl^X}&PD`@WRZC$>8I+2YV_' \
			  b'E;&J?A|z%u4~%xH4QWus&2gUHJV$+=I{}8+#h$W>80a-T07@4I*+sWB*xM8p84@Ells5^sz++*J-xJdR4et<`_wl}=sI>^ckz6hbB' \
			  b'~r!xtr$bz7vjo6ejYso?C9Ce$UOh>j1UyoxWn-EktGy=30Im=RbQq&8a8r|8y^%&z8JaD@C8U@N_Gc!?T~Zw!}LpzIq(J58L)8Cev' \
			  b'}%ku%Pt_I-ytAEt6w-JxA{zFS_&E2VnT5hY3Gj6E0DP`NES!P@fG*&D}GShQro(=-m_UOsX)wSRE?cR=k6gT0HXe(kWCv#kB|&3V;' \
			  b'SZu(@#?X-PyRdL+9KhKtaN$oFQ(epW#ZDWTlr*ix1iEmi*S~ASq0zd8#4x-mhzGv(ns=sh**GtxUv2<1ewO_U6Qp>)fE_eskS8mN(' \
			  b'>pB<>SX+4C?!zf64}Je-1(gqMhbgq*mEZ2Hq-WAYJ9TUR@wZ}Y=>7O%_nhr?UFCNTypQ&q_VzA~#{KbF$wu1mqQ!4n>v^iB+VV5C?' \
			  b'yt+~dfMMPIGN^i=e*Zr^!iP&6<YDw`0^)}wBPz#eZ6%)-ZV5t`>*sj{EfEX{LH$qY5&84QTypxIe6d0)>fTz==IT7yxwoUz}mOdO*' \
			  b'priZ`2udqn;OqnRKHLrW^JA_!#oInw4)I>s}ue+Te%THUy!_1_g?3P@%*I4N7eY!yF5IUYKix56Wx^z&slih}fV)xeXzxut9@L8^T' \
			  b'a!f!_<&HUwe54Izlyph1lV0WZ|r;D-e^1fk9b1?p|kpuvKm7Z%#!heb9hu-FC_F0dgCjTR_gxX^|mG})lSPb^Tqpj!~~L$f_+Kg8^' \
			  b'b`JlxHKNvOypw$Ki+H6oEZbKLn_6&Wn#D)O0+n_?yh7fev5QdaJLm#AV2tua~ApjdRKzph_$k-47(*^~y_T*JqYC{;h>{V#6%&IuN' \
			  b'zECI}3MtUdRZUUIl$Oh_A^54SdcCUV)fBi$Zq=YiR>N?yq-tJRA*(*PgsWbEK=Z>&p(+8A7?;YO{cxGA2H<j84Z<omt6)%5NKUSnT' \
			  b'NPNtTQ#3XvhfPu>I;TR!mMSjs_NHB0<4p+4SV5A)*4bnVKR;Na%%vtl2rw+W}UTgNFf<`jocc74NUcU{hAVjYvooAu9MX;Trb~+AZ' \
			  b'%o6P*K8K7;fn6tdV{CGg(#OMp+HPO-v;{Lu7aN%B>n~VsrL~gF$i!Z<a%+!e*wb{;(1Zz%6oX2yW$}^QvKz*tf~8VYpphkq_>WRX^' \
			  b'M*t3kL+R#mv0sbtD3IWB(ARZR(!NN(YgR1_as;5|(B`@$Lt`@KvJ_!TV_fcxb2`rv+94Zs7k8iZfSstUi9L#M!la%&hKl3K~R_i)2' \
			  b'7>$LRJp9M9<*RLEFKRn{F1mRJKMS;g078M?MSTy*x!xDzA{VilZPdF?-c+z1Bz*7#30^1xG6}CGpA^45MqQTP+OBi<aw~+IEr^6D2' \
			  b'XB?IgJnOJ%u&cj?ocYf=EPi<2VF|)+hed%G92N~;>~A3th&>LAA6{};6nNQTQQ;MbB@DmqZy`^UR~?ohyymc|u(!X3JZN6;Zy`^de' \
			  b'U3Q$;SEQ`eDFJm#Sd>fECJZ>uqg1B!=l354oeu`am3IE?>a02c+X)`;eCfC1P2_JFnr*Mp$|TESb}iSVF|%U4vPjKJEH1?PaKv2eC' \
			  b'n_$@R=j>Dtzv+gyHv&d!fM>yk0uQ>!mMwz4R5Ym%irp(l@+b`U9_*zUB4OVO}r&53iTL<Mq-XdA;;Kub2MB>!ttY_0pety>x`vOMl' \
			  b'_@(qDPK^aHP#{>JO2A9=m>cU~|3gVjq&-_KdegaZWoF;}pHM+sJOpx`hb%~|o{Ai+T#ELg=MoK-Il<s9;39%r8)hjI4%@EE~<94<J' \
			  b'3BLpjWtY8&K3J&9OoC7{QUT^?M3085m;1G@x9L5vGx_o$|;2@3_9KvyeH5@P2=);o)2XKO51t*HNt2jw;7*7^)(9pxVupCbj9Kp$g' \
			  b'7vZUbqj(zUqC%V^xC&1fT#HjV7nfkZ;37PO^PECFlk?m%Jd1N#F-{X)j%N$5!~(&Ecn;@C4SG3O6r)dYCHe)|U_fv!202&Op(1!5s' \
			  b')8#pB)Awg&Q%c%bFMDL>6~kd@La*gID_+oT0D<)T`kV!TwjOh3tot`L~LuYP>dhN*_=zuu}G|UAr^BkFT@hDt`aO2T!?c7m*HH&Ra' \
			  b'hq0Rf6*b&&7z~g;*}Q2rI<e=VGPcC{}TfmSHvL+De=+xB;VL?S)t))?0|Rg3EA$$cF`3C+1m=_2Swxq`$&xXrb{*>dktG?s4-67fR' \
			  b'kT?e04B;3DA-6U);nm)Ewq&s)2m%IFzay<Nb)73s-DYZ_At7ng^P!dp6-M!hBN>e?>s^Oi~(F*6a%n(ldP;@*mlq~4{Y;p*CcBD^I' \
			  b'sCMII0C*A5HQO$NV|M%}8>fGBLBaLkdmv=xj_coWrF_Cb2cgMK5vKBp__PFe=MR-dl6RC8<-Q8^nZ>g4q?)J&mD!iq;Tt1VDw>IuA' \
			  b';`%8`yv4b<3LU$XdaA`$ZwcWoK~Anr+H^CVxTMcpCPV-0>~wF_VY~2_PMNOfWRiQUG?wZuZdVE&ecrk}^9;-w-7b1d$=)iAWr>)J#' \
			  b'}}rBw=|k@PqE(aVp?J+_g2-OB*(J51tkLaR&AtGneOCL-NiZxk$bC2XGtWBbWD?HflK}{)8CuR_s!(q7MOb6^=Ckqd#f|BgB&OBdR' \
			  b'xl9)f*YpQ<or>kb7x@UBX*;x+9Tlb9Zga#9N2;?T4Q7)aV^)w{o&u?6<6eX4(^V^XS6m+*>Jm=}5Tz5Wt^`y2MDC*;sqGtKKf+-pb' \
			  b'S2M8@SB4}1EmphU`Wdv{;l=Pi@bWASXpFip1#JzUXO1;yftHnIcUT-zmb&6Uogt8Y2DvaeETMz?=sf|ts5$y~joGZRm^_BxE0$=(*' \
			  b'GvE9|RUEaS6a{1Z4N>oAdgxjyC9$YQFq0yF3x$JEX_ZE$(TMSQG#?7}Xyn=hHHRu~v#&GlW!nJ+gmbv|E>cMrQ3Uaq3vAta>>Jpvo' \
			  b'>r|WRW~ShJ;VqG&-|QybOiR2<{GKbTceuXauI~H&WCy8Uy9_-^j+At($BY{u>wGaV<*CrodaT<U<o}&ac#WvcQwh^(@zj{QX=GR#`' \
			  b'=5^fFWxrr|2N?E;<XZAC-L<XZ<P23iGL>XjS}A^aj(RiB)(bV%@W@t@vRcyCh_eO-y!jx65l29-4g#?;w=*2Bk{cw-zV|?5<ejEFC' \
			  b'_k@#1BgRki-v5{40qck@!)GACvfTiGMBeR*9dG_(_SMl6afM+a>;u#7|4SL*kthKO^z867Q1uIf<W_c(=qaNc<vU!^AyAG#ich5|R' \
			  b'1w!SNwI__BrkOLu&QsPPeN@V7)p@m1PLzhz#9uhE7vYa4N|b-}p!crxDWbUf>0XxwLA@CNN=jr+U)<G#tp-EVb!i;X*Y6~4{KeaE`' \
			  b'sr1*sR#Q3E6$#GBo6yo9CJ`e9%m%dL|(t`)+;^+sT-U45+z=t2uZ>sS_I)?S-zrM`K;6Xa}Agfz5{Zot#enc#d_%ZF)gP&NvY3ouw' \
			  b'nI-){rTsPQi>8h?Y$BIr@H1kIHD?n^Gm*;VWs-@Qk%ylXV<ab%8yT<$e{W6e3-WKa2M^I9mssP>TtY8hVs*`p45eMaw7PypyBgVyi' \
			  b'C@#jS<`JYyF2k4x~8G*uQ{y0K*2w7Q&Wdw>eP<kT1z-g2Wc+&Cvbvda<?)700'

	correct_df = pickle.loads(gzip.decompress(base64.b85decode(correct)))
	pandas.testing.assert_frame_equal(correct_df, df)

	df_o = q.load_dataframe(preserve_order=True, strip_whitespace=False)
	pandas.testing.assert_frame_equal(correct_df[[
		'STATEFP',
		'STATENS',
		'AFFGEOID',
		'GEOID',
		'STUSPS',
		'NAME',
		'LSAD',
		'ALAND',
		'AWATER',
	]], df_o)

	df_s = q.load_dataframe(preserve_order=False, strip_whitespace=True)
	correct_df.NAME = correct_df.NAME.str.strip()
	pandas.testing.assert_frame_equal(correct_df, df_s)

def test_repeated_splitting():
	df = pandas.read_csv(example_file("MTCwork.csv.gz"))
	df.set_index(['casenum', 'altnum'], inplace=True)

	dfs = DataFrames(df, crack=False)
	d1, d2 = dfs.split([80, 20])
	assert d1.n_cases == 4024
	assert d2.n_cases == 1005
	d11, d12 = d1.split([50, 50])
	assert d11.n_cases == 2012
	assert d12.n_cases == 2012

	dfs = DataFrames(df, crack=False)
	d1, d2 = dfs.split([80, 20], method='shuffle')
	assert d1.n_cases == 4024
	assert d2.n_cases == 1005
	d11, d12 = d1.split([50, 50])
	assert d11.n_cases == 2012
	assert d12.n_cases == 2012

	dfs = DataFrames(df, crack=True)
	d1, d2 = dfs.split([80, 20])
	assert d1.n_cases == 4024
	assert d2.n_cases == 1005
	d11, d12 = d1.split([50, 50])
	assert d11.n_cases == 2012
	assert d12.n_cases == 2012

	dfs = DataFrames(df, crack=True)
	d1, d2 = dfs.split([80, 20], method='shuffle')
	assert d1.n_cases == 4024
	assert d2.n_cases == 1005
	d11, d12 = d1.split([50, 50])
	assert d11.n_cases == 2012
	assert d12.n_cases == 2012


def test_co_only():

	x_co = pandas.DataFrame(
		numpy.random.random([20, 3]),
		columns=['Aa', 'Bb', 'Cc'],
	)
	x_co.index.name = 'caseid'

	with raises(ValueError):
		# missing altcodes
		DataFrames(
			co=x_co,
			av=True,
		)

	with raises(ValueError):
		# trying to use ca
		DataFrames(
			ca=x_co,
			av=True,
		)

	d = DataFrames(
		co=x_co,
		av=True,
		alt_codes=[1, 2, 3, 4, 5, 6],
	)
	assert d.n_alts == 6
	assert d.n_cases == 20
	assert d.data_av.shape == (20, 6)
	all(d.data_av.dtypes == numpy.int8)
	all(d.data_av == 1)


def test_ca_initialization():
	cax = pandas.DataFrame({
		'caseid': [1, 1, 1, 2, 2, 2],
		'altid_bad': ['aa', 'bb', 'cc', 'aa', 'bb', 'cc'],
		'altid_good': [1, 2, 3, 1, 2, 3],
		'altid_str': ['1', '2', '3', '1', '2', '3'],
		'buggers': [1.2, 3.4, 5.6, 7.8, 9.0, 5.5],
		'baggers': [22, 33, 44, 55, 66, 77],
	})

	import pytest

	with pytest.raises(ValueError):
		d = DataFrames(
			ca=cax.set_index(['caseid', 'altid_bad']),
		)

	d = DataFrames(
		ca=cax.set_index(['caseid', 'altid_good']),
	)
	assert all(d.alternative_codes() == [1, 2, 3])

	d = DataFrames(
		ca=cax.set_index(['caseid', 'altid_str']),
	)
	assert all(d.alternative_codes() == [1, 2, 3])


def test_dfs_init_ca():
	from larch.data_warehouse import example_file

	df = pandas.read_csv(
		example_file("MTCwork.csv.gz"),
		index_col=['casenum', 'altnum']
	)

	d0 = DataFrames(ca=df, crack=True, ch='chose', wt_name='wgt')
	assert d0.data_wt is not None
	assert d0.data_wt.columns == 'wgt'
	assert d0.data_ch is not None
	assert d0.data_ch.shape == (5029, 6)
	assert d0.data_av is not None
	assert d0.data_av.shape == (5029, 6)

	d1 = DataFrames(ca=df, crack=True)
	assert d1.data_wt is None
	assert d1.data_ch is None
	assert d1.data_av is not None
	assert d1.data_av.shape == (5029, 6)

	d2 = DataFrames(df)
	assert d2.data_wt is None
	assert d2.data_ch is None
	assert d2.data_av is not None
	assert d2.data_av.shape == (5029, 6)
	assert d2.data_co is None
	assert d2.data_ca is None
	assert d2.data_ce is not None
	assert d2.data_ce.shape == (22033, 36)

	d3 = DataFrames(ca=df, crack=True, ch=df.chose, wt_name='wgt')
	assert d3.data_wt is not None
	assert d3.data_wt.columns == 'wgt'
	assert d3.data_ch is not None
	assert d3.data_ch.shape == (5029, 6)
	assert d3.data_av is not None
	assert d3.data_av.shape == (5029, 6)
	assert pandas.isna(d3.data_ch).sum().sum() == 0

	d4 = DataFrames(ca=df, crack=True, ch=df.chose, wt='wgt')
	assert d4.data_wt is not None
	assert d4.data_wt.columns == 'wgt'
	assert d4.data_ch is not None
	assert d4.data_ch.shape == (5029, 6)
	assert d4.data_av is not None
	assert d4.data_av.shape == (5029, 6)

	d5 = DataFrames(ca=df, crack=True, ch=df.chose, wt=df.wgt)
	assert d5.data_wt is not None
	assert d5.data_wt.columns == 'wgt'
	assert d5.data_ch is not None
	assert d5.data_ch.shape == (5029, 6)
	assert d5.data_av is not None
	assert d5.data_av.shape == (5029, 6)
	assert d5.data_co.shape == (5029, 31)
	assert d5.data_ca is None
	assert d5.data_ce is not None
	assert d5.data_ce.shape == (22033, 5)

	with raises(ValueError):
		bad = DataFrames(co=df)

def test_dfs_init_co():
	from larch.data_warehouse import example_file
	raw_data = pandas.read_csv(example_file('swissmetro.csv.gz'))
	keep = raw_data.eval("PURPOSE in (1,3) and CHOICE != 0")
	selected_data = raw_data[keep]

	d0 = DataFrames(selected_data, alt_codes=[1, 2, 3])
	assert d0.data_co.shape == (6768, 28)
	assert d0.data_ca is None
	assert d0.data_ce is None
	assert d0.data_ch is None
	assert d0.data_av is None

	d1 = DataFrames(co=selected_data, alt_codes=[1, 2, 3])
	assert d1.data_co.shape == (6768, 28)
	assert d1.data_ca is None
	assert d1.data_ce is None
	assert d1.data_ch is None
	assert d1.data_av is None

	with raises(ValueError):
		DataFrames(ca=selected_data, alt_codes=[1, 2, 3])

	d2 = DataFrames(co=selected_data, alt_codes=[1, 2, 3], ch='CHOICE')
	assert d2.data_co.shape == (6768, 28)
	assert d2.data_ca is None
	assert d2.data_ce is None
	assert d2.data_ch is not None
	assert d2.data_ch.shape == (6768, 3)
	assert all(d2.data_ch.sum() == [908, 4090, 1770])
	assert d2.data_av is None

	d2 = DataFrames(co=selected_data, alt_codes=[1, 2, 3], ch=selected_data.CHOICE)
	assert d2.data_co.shape == (6768, 28)
	assert d2.data_ca is None
	assert d2.data_ce is None
	assert d2.data_ch is not None
	assert d2.data_ch.shape == (6768, 3)
	assert all(d2.data_ch.sum() == [908, 4090, 1770])
	assert d2.data_av is None

	d2 = DataFrames(co=selected_data, alt_codes=[1, 2, 3], ch='CHOICE', wt='GROUP')
	assert d2.data_co.shape == (6768, 28)
	assert d2.data_ca is None
	assert d2.data_ce is None
	assert d2.data_ch is not None
	assert d2.data_ch.shape == (6768, 3)
	assert all(d2.data_ch.sum() == [908, 4090, 1770])
	assert d2.data_av is None
	assert d2.data_wt is not None
	assert d2.data_wt.shape == (6768, 1)

	d2 = DataFrames(co=selected_data, alt_codes=[1, 2, 3], ch='CHOICE', wt=selected_data.GROUP)
	assert d2.data_co.shape == (6768, 28)
	assert d2.data_ca is None
	assert d2.data_ce is None
	assert d2.data_ch is not None
	assert d2.data_ch.shape == (6768, 3)
	assert all(d2.data_ch.sum() == [908, 4090, 1770])
	assert d2.data_av is None
	assert d2.data_wt is not None
	assert d2.data_wt.shape == (6768, 1)