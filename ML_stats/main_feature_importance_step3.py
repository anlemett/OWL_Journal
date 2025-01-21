import warnings
warnings.filterwarnings('ignore')

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import csv

DATA_DIR = os.path.join("..", "..")
DATA_DIR = os.path.join(DATA_DIR, "Data")
ML_DIR = os.path.join(DATA_DIR, "MLInput")
FIG_DIR = os.path.join(".", "Figures")

RANDOM_STATE = 0
PLOT = False

CHS = True
BINARY = False

#MODEL = "SVC"
MODEL = "KNN"

N_ITER = 100
N_SPLIT = 10
SCORING = 'f1_macro'
#SCORING = 'accuracy'

#TIME_INTERVAL_DURATION = 300
TIME_INTERVAL_DURATION = 180
#TIME_INTERVAL_DURATION = 60
#TIME_INTERVAL_DURATION = 30
#TIME_INTERVAL_DURATION = 10
#TIME_INTERVAL_DURATION = 1

SAVE_COORD = True

def find_elbow_point(x, y):
    # Create a line between the first and last point
    line = np.array([x, y])
    point1 = line[:, 0]
    point2 = line[:, -1]

    # Calculate the distances
    #print("point1:", point1)
    #print("point2:", point2)
    #print("line:", line)
    distances = np.cross(point2-point1, point1-line.T)/np.linalg.norm(point2-point1)
    elbow_index = np.argmax(np.abs(distances))

    #return x[elbow_index]
    return elbow_index


##############
def main():
    '''
    #64 features
    if MODEL == "SVC":
        if BINARY:
            bal_accuracies = [0.8088313061872026, 0.8196377578001057, 0.8180248545742993, 0.8180248545742993, 0.7904309888947647, 0.7896245372818614, 0.7896245372818614, 0.7996245372818614, 0.7996245372818614, 0.7996245372818614, 0.7995438921205712, 0.8011832363828664, 0.7887374405076679, 0.7927961396086727, 0.7927961396086727, 0.7927961396086727, 0.7879309888947648, 0.7927961396086726, 0.7919896879957694, 0.7936832363828662, 0.7819896879957694, 0.790376784769963, 0.7987638815441565, 0.7795703331570597, 0.7988313061872026, 0.7763313061872026, 0.7888313061872025, 0.7771377578001059, 0.7771377578001059, 0.7762506610259122, 0.7987374405076679, 0.8063313061872025, 0.7971113167636172, 0.7971113167636172, 0.7838855103120042, 0.7946919619249074, 0.7862506610259122, 0.7787638815441565, 0.7703635642517186, 0.7695571126388154, 0.7671509783183501, 0.776344526705447, 0.7763445267054468, 0.7579574299312534, 0.7463445267054469, 0.7387638815441566, 0.7179574299312533, 0.7471509783183501, 0.7904574299312535, 0.7728767847699629, 0.7904574299312532, 0.7863445267054469, 0.7848122686409307, 0.7856451612903226, 0.795551295610788, 0.7747448439978847, 0.7714925965097832, 0.7650409836065574, 0.6535219460602856, 0.7806451612903226, 0.7441538868323637, 0.6602432575356953, 0.6139661554732945, 0.6187109994711794]
            f1_scores = [0.7961473707543458, 0.8092165591519777, 0.8022891437590394, 0.8022891437590394, 0.7830628175602381, 0.7793096521597316, 0.7793096521597316, 0.7915958117022888, 0.7915958117022888, 0.7921949329353384, 0.8044238585829924, 0.8077831922688729, 0.7877824341875688, 0.8017985724532048, 0.8017985724532048, 0.8017918684524951, 0.7856151349135946, 0.800393102571911, 0.7971957076947365, 0.792500948670894, 0.7881272846032129, 0.793275068181078, 0.7958495979594418, 0.7792193622225632, 0.7835533481374065, 0.7656899356218637, 0.7741569803925687, 0.7696308540262666, 0.7700449495815858, 0.7807392724996197, 0.7986033700862214, 0.7979191705094033, 0.7975120294182786, 0.7975120294182786, 0.7734657035131478, 0.7857273924902256, 0.7876853072845426, 0.7774961214111905, 0.7805238707090303, 0.7763228238550446, 0.7662671362475034, 0.768500726621572, 0.7644470379555137, 0.7532066168122122, 0.7434451360171845, 0.7472634800200794, 0.7175729901746621, 0.7543269533953338, 0.7846569427119066, 0.7812418319460781, 0.7843808897188055, 0.778384152151177, 0.7619180319758998, 0.7721188035783845, 0.7910956161188548, 0.7650909616251157, 0.7578801068554346, 0.7282628157378277, 0.6436824860173822, 0.7598991981624241, 0.7236662414442085, 0.6084359927059587, 0.5819076930459911, 0.5898572703605574]
        else:
            bal_accuracies = [0.7188307915758896, 0.727922502334267, 0.7181605975723623, 0.7169260296711277, 0.7295450772901754, 0.7123804336549434, 0.7148054777466541, 0.726405747484179, 0.7140808175121901, 0.7200051872600893, 0.7133105093889409, 0.7090968980184666, 0.720187260089221, 0.7183546010996992, 0.7130444029463637, 0.7173140367258015, 0.7137747691669261, 0.7223742089428364, 0.7245891690009338, 0.7024141508455234, 0.6985563855171699, 0.7189926340906733, 0.7134505654113497, 0.697443718228032, 0.6896265172735762, 0.7048848428260193, 0.7208475982985788, 0.7111375661375663, 0.7277642909015458, 0.7171262579105717, 0.7373643531486669, 0.7379463637306775, 0.7276159352629941, 0.7351291627762216, 0.7308035065878202, 0.7294926859632742, 0.7456172839506173, 0.7328140885984024, 0.7219498910675382, 0.7535387488328664, 0.7413678804855276, 0.7237706193588547, 0.7422523083307397, 0.7549714700695093, 0.7628950098557942, 0.7553698516443615, 0.7651976346093993, 0.7562558356676005, 0.7705197634609399, 0.7718425147836914, 0.7553153854134247, 0.7569140989729225, 0.7762221184770205, 0.7370759414877062, 0.7219275858491545, 0.7277176055607428, 0.7299891067538127, 0.7289724037763253, 0.6646519348480133, 0.677376802572881, 0.6546140678493619, 0.5461028114949684, 0.5096405228758171, 0.38630044610436765]
            f1_scores = [0.7341251690696968, 0.7365699526202183, 0.733296038180596, 0.7321634465816586, 0.738450855940131, 0.7312882569172594, 0.732561909419745, 0.7399696059014556, 0.7305451729287656, 0.7345435542996219, 0.7255925826133561, 0.726241200156279, 0.7288713092387467, 0.7269160915935045, 0.7216099779157492, 0.7266443819940909, 0.7225536383905373, 0.7319025716326893, 0.7356014658196082, 0.7130097382268036, 0.7069130601839666, 0.7298167227713009, 0.7332224595043775, 0.7080248054276834, 0.6990288615860707, 0.7139863257693713, 0.7307916330406685, 0.7197975191343764, 0.735308322888591, 0.726110561240415, 0.7480081791791664, 0.7434906907302479, 0.7324292704665686, 0.7442221343010827, 0.7391622469279426, 0.7365910426566924, 0.7547291615339298, 0.7416246544593145, 0.7332133507060572, 0.7567458332573217, 0.7517580732854403, 0.7346600254926443, 0.753708668806956, 0.7650021413255416, 0.7730458821004517, 0.7656359177785143, 0.769695529259434, 0.7577598165911456, 0.7703894785685378, 0.7730332571834421, 0.7503174549178487, 0.7543029107276666, 0.780291813179012, 0.7249067311902092, 0.7164804357252872, 0.7316680441552398, 0.7235950913726809, 0.7430259006370417, 0.6651730630932078, 0.6515258542353065, 0.6404208276955268, 0.5126633938922505, 0.4425276169734385, 0.3761624238171006]
    elif MODEL == "KNN":
        if BINARY:
            bal_accuracies = [0.7595703331570597, 0.7603900052882073, 0.7503900052882073, 0.7645029085140138, 0.765309360126917, 0.7636158117398203, 0.7744896879957694, 0.7852961396086726, 0.7952961396086726, 0.7719896879957695, 0.781170015864622, 0.7703767847699629, 0.7703767847699631, 0.7828767847699629, 0.7828767847699629, 0.7828767847699629, 0.7803900052882072, 0.7736964569011106, 0.7503635642517187, 0.7703900052882073, 0.7503900052882073, 0.775309360126917, 0.7836158117398202, 0.792809360126917, 0.7936158117398202, 0.8145029085140137, 0.8161158117398202, 0.792809360126917, 0.750376784769963, 0.7645029085140138, 0.7661158117398202, 0.7861290322580646, 0.7853225806451614, 0.7853225806451614, 0.7961290322580645, 0.7969354838709678, 0.7861964569011105, 0.7928900052882073, 0.775309360126917, 0.7836832363828662, 0.8093283976731888, 0.7969222633527234, 0.8294354838709678, 0.7886832363828662, 0.8109677419354838, 0.7669222633527235, 0.7885351665785298, 0.8152419354838709, 0.8193548387096774, 0.80853516657853, 0.7985351665785299, 0.8185351665785298, 0.8169354838709678, 0.8077154944473822, 0.7111290322580645, 0.7953225806451614, 0.8185483870967742, 0.7687638815441564, 0.7477287149656267, 0.741170015864622, 0.7194354838709677, 0.6979574299312533, 0.6152154944473822, 0.5620029085140137]
            f1_scores = [0.7789349516559503, 0.7855782294794549, 0.7732920699368977, 0.7927332967763707, 0.7878842594599925, 0.7956862741417033, 0.7949756587462844, 0.8114562520198769, 0.8187943472579722, 0.7995321327786845, 0.8042215097263181, 0.7909814891091432, 0.7898586083336036, 0.7985321946635091, 0.7985321946635091, 0.7985321946635091, 0.7971435655375865, 0.7890794331590218, 0.7673422936397916, 0.7904150639287973, 0.7729164078784507, 0.7988340498530986, 0.8087975348323049, 0.8154818713341074, 0.8212454658041534, 0.829207049315082, 0.8302131551694973, 0.8093862614873423, 0.7640027020079299, 0.7827190849677288, 0.7869631834459538, 0.8080012776796532, 0.8069691906587618, 0.8070367275044056, 0.8164350737335585, 0.8187810121089086, 0.7973308269298195, 0.8054813817618696, 0.7936346187971304, 0.7979556181895812, 0.8418184584383382, 0.8220946860402709, 0.8477767383311203, 0.804877719725839, 0.8490529883437341, 0.7995284410795479, 0.8218051198143781, 0.8404209231710343, 0.8374465637702903, 0.8374009588701815, 0.8219609077906929, 0.8359544818673068, 0.8324931360954333, 0.8316845281044725, 0.7400525493417301, 0.8165543577759454, 0.8423926025180801, 0.773463588225744, 0.7787276069149892, 0.7618996921192147, 0.748354426267843, 0.7096000048552413, 0.6413070844166329, 0.5698331858448967]
        else:
            bal_accuracies = [0.7347375246394854, 0.7365323166303559, 0.7573347857661583, 0.7419083929868243, 0.723429297644984, 0.7407661583151779, 0.7459783172528269, 0.7312864405021269, 0.7420126569146177, 0.7540071584189232, 0.7330267662620604, 0.7478664799253034, 0.753022616453989, 0.7612060379707438, 0.7605477746654217, 0.769627554725594, 0.7662942213922606, 0.7599538333852058, 0.7638515406162465, 0.7683312584292976, 0.7608341114223468, 0.7586310820624547, 0.7639910779126466, 0.7491850814399835, 0.7572964000414981, 0.7596052495072103, 0.7499050731403673, 0.7528384687208216, 0.7755752671438946, 0.7755752671438946, 0.7811396410416018, 0.7764939309056957, 0.7679240585122937, 0.7622642390289449, 0.7716599232285507, 0.7313616557734205, 0.7542426600269737, 0.7367335823218176, 0.7587306774561676, 0.7369135802469137, 0.7364213092644465, 0.7241923436041082, 0.7377518414773316, 0.7790984541964934, 0.7752889303869697, 0.7541306152090466, 0.7359134765017118, 0.7533328146073244, 0.7396130303973442, 0.7210177404295052, 0.670552961925511, 0.6758590102707751, 0.7162397551613238, 0.7195331465919701, 0.698269011308227, 0.7215779645191409, 0.7392909015458036, 0.7231989832970225, 0.6929499948127399, 0.6964462081128747, 0.6522678701110073, 0.5994926859632741, 0.5041041601825916, 0.44648978109762416]
            f1_scores = [0.74295647692683, 0.7421516294754824, 0.7650471057356018, 0.7561738553882988, 0.72507330109144, 0.7523031427065443, 0.7469659610758321, 0.7439212845203779, 0.7547305787286203, 0.768394434240496, 0.7418228466288653, 0.7579149833359407, 0.7630591614491983, 0.7707511174226822, 0.7675388945011635, 0.7781140735755935, 0.7748349253886134, 0.7707603843895873, 0.7712304506998255, 0.7745040963139547, 0.7747694792823554, 0.770840722618436, 0.7770668930657048, 0.7547677953277525, 0.7679726708560798, 0.7657150972029378, 0.7539985058672479, 0.7614271103904334, 0.769830449158092, 0.769830449158092, 0.7772226684697795, 0.7770284024344203, 0.7724705951992925, 0.7644657000583182, 0.7729426313182401, 0.7424181654347415, 0.760149032340627, 0.7445584878443421, 0.7633317975599151, 0.734939272234078, 0.7357826381254536, 0.7236095550721793, 0.7408742743404744, 0.7844224053277508, 0.778025286956284, 0.7570442530219659, 0.745154688099602, 0.768648843207785, 0.7515258441088387, 0.7331113492599497, 0.6790904490447877, 0.6888917120582166, 0.7255121777060707, 0.7164903507670252, 0.7094400479585059, 0.7393637561975448, 0.7585968413767595, 0.7429822850394846, 0.7088471680168797, 0.7101334220071867, 0.6638488537406068, 0.6024846016576861, 0.5039147128383382, 0.42947303853531044]
    '''
    
    # init_blinks_no_head
    if MODEL == "SVC":
        if BINARY:
            bal_accuracies = [0.7287638815441565, 0.7063445267054469, 0.7215058170280276, 0.7079574299312533, 0.7079574299312533, 0.713118720253834, 0.7079574299312533, 0.7712638815441565, 0.7287638815441564, 0.7195571126388155, 0.7062506610259123, 0.7062506610259123, 0.7146377578001057, 0.7338313061872026, 0.7287506610259122, 0.7187638815441566, 0.7045703331570599, 0.7102961396086727, 0.7169896879957693, 0.7194896879957694, 0.7171509783183502, 0.7447316234796404, 0.7514793759915389, 0.7214119513484929, 0.7438313061872026, 0.74068614489688, 0.7014119513484929, 0.7189925965097832, 0.7062638815441566, 0.700618720253834, 0.6838445267054468, 0.7197990481226864, 0.7446377578001057, 0.7738180856689583, 0.7662638815441565, 0.7514251718667373, 0.7698796932839767, 0.7282258064516128, 0.7650542041248016, 0.7407125859333686, 0.7190864621893178, 0.7551480698043365, 0.7115322580645161, 0.7622316234796405, 0.7591671073506082, 0.7294500264410365, 0.724892913802221, 0.7245452141723956, 0.622181385510312]
            f1_scores = [0.7264247811031261, 0.6949674808895709, 0.6878001231660704, 0.6979853663025715, 0.6979853663025715, 0.6852121296299041, 0.6979853663025715, 0.7635129032132866, 0.7236067546949132, 0.7180513185247591, 0.7059205133154266, 0.7059205133154266, 0.7132985029814045, 0.7215499872017553, 0.7231970360046905, 0.7114631442622292, 0.7030672093628674, 0.7226811069376808, 0.7195504287448701, 0.7259879992345255, 0.6934122209555367, 0.73876595108825, 0.737552211664975, 0.7057421397723708, 0.7310651202427064, 0.7176937787518839, 0.691505843720952, 0.6990530912525149, 0.7011591112624609, 0.681339449054736, 0.6827735834122726, 0.7058858672071665, 0.7394250177515478, 0.7673463567439562, 0.7727802692815253, 0.7373870917202152, 0.7460005691790963, 0.7189721318710872, 0.7297866279918233, 0.7132236938190402, 0.6969975530617082, 0.7169739305493955, 0.7042273583303782, 0.7543238824265568, 0.7306826339529835, 0.6806879944968784, 0.6634009349372808, 0.5866989079999494, 0.49900220062994893]
        else:
            bal_accuracies = []
            f1_scores = []
    elif MODEL == "KNN":
        if BINARY:
            bal_accuracies = [0.7354706504494978, 0.7495835536753042, 0.748777102062401, 0.7695835536753041, 0.7571377578001058, 0.7795703331570598, 0.780376784769963, 0.792876784769963, 0.7812638815441565, 0.7688180856689583, 0.7763313061872025, 0.7805248545742993, 0.7530248545742995, 0.7663180856689582, 0.788011634056055, 0.7988180856689582, 0.7855248545742993, 0.8063313061872025, 0.8072184029613961, 0.7838987308302486, 0.7695438921205712, 0.7572051824431518, 0.798011634056055, 0.7572448439978847, 0.7771919619249075, 0.7588180856689581, 0.7371245372818613, 0.7496245372818614, 0.7687903225806452, 0.7905248545742992, 0.7663313061872026, 0.7880248545742994, 0.7330248545742993, 0.7662771020624009, 0.7403900052882074, 0.7787771020624008, 0.7536158117398202, 0.7552287149656267, 0.7279574299312535, 0.7261832363828662, 0.752876784769963, 0.742876784769963, 0.7620703331570597, 0.6910483870967743, 0.7109413008989953, 0.7179574299312533, 0.6586964569011104, 0.5749061343204653, 0.5070703331570598]
            f1_scores = [0.7464930473857259, 0.7678512413060161, 0.7632104800368115, 0.7846750948258165, 0.7666736858655911, 0.7939304723031858, 0.7979149614568332, 0.8036028922382485, 0.7860470170105367, 0.7637865727349326, 0.7800374425835439, 0.7753823504950512, 0.7540364422372067, 0.7688075759325392, 0.7801641263015091, 0.7957130338697614, 0.7795775406022483, 0.8002396539238681, 0.7935307681232293, 0.7745016800409961, 0.7870965113562577, 0.7477492609895566, 0.7883752057775866, 0.7409190792087196, 0.7683070608588276, 0.7546023067467431, 0.7473579026493697, 0.7529678698270909, 0.7737494357944547, 0.7788752862911968, 0.7668139590704675, 0.77972814487592, 0.7400215493407541, 0.77577266256314, 0.7660724442165513, 0.7851485327634004, 0.7885888903733733, 0.7894467046457903, 0.7355234452000593, 0.7437067093525385, 0.7592826415849713, 0.75483179446851, 0.7803865490917542, 0.7273764470938746, 0.7484960724043715, 0.7282758254002196, 0.6900020126516444, 0.5880573812927599, 0.49568094966377635]
        else:
            bal_accuracies = [0.7227061935885466, 0.7071340388007055, 0.6875619877580663, 0.7043744164332401, 0.7201825915551405, 0.7255591866376181, 0.6664436144828301, 0.6878721859114016, 0.6962734723519037, 0.7010913995227721, 0.7071122523083307, 0.7199253034547153, 0.7156411453470277, 0.6941466957153232, 0.7040247951032264, 0.6944107272538645, 0.6967553688141924, 0.6844065774457933, 0.6861012553169417, 0.6910011411972197, 0.7129178338001867, 0.6890185703911194, 0.6854030501089324, 0.6910950306048346, 0.6994745305529619, 0.7056312895528581, 0.7326641767818238, 0.7149745824255629, 0.7182461873638344, 0.7029738562091503, 0.6958387799564271, 0.698622263720303, 0.7154227616972715, 0.6775236020334059, 0.7110893246187364, 0.7034204793028322, 0.6868056852370578, 0.6596586782861292, 0.6290403568834941, 0.6300207490403568, 0.6281165058616038, 0.663256043158004, 0.6623358232181762, 0.6419109866168691, 0.6283260711692085, 0.647190061209669, 0.6059228135698724, 0.5630926444651936, 0.397050523913269]
            f1_scores = [0.7265092146285592, 0.7218622733723778, 0.6999999798088239, 0.7190675424098957, 0.7288612325672982, 0.730013970178094, 0.6870927646589254, 0.7043460140581992, 0.7095459542400236, 0.7098463033701423, 0.7220971057754488, 0.7326960992762548, 0.7309455801201512, 0.7079603886821615, 0.7228658216014672, 0.7192976718170845, 0.7173084164261444, 0.707209532729858, 0.7090601637498872, 0.7079105972901537, 0.7293225720362116, 0.7111688981925325, 0.7181344581205237, 0.7116477877098399, 0.7288374107261978, 0.732963056724991, 0.7618373161839032, 0.7343163728302541, 0.7424722163884929, 0.7248184001590446, 0.7189750445839859, 0.7250402387049245, 0.741900348045501, 0.6957060015909992, 0.7275887409371575, 0.7224954700654121, 0.7028191511921427, 0.6694830931166389, 0.6437325323801236, 0.6451675631084477, 0.6496711297963538, 0.6843848761315983, 0.685302449488745, 0.6542437082974654, 0.6500084338002704, 0.6582627375214156, 0.6179187816937025, 0.5803217997341624, 0.3911018283777896]
    
    '''
    #15 features
    if MODEL == "SVC":
        if BINARY:
            #threshold=0.7
            bal_accuracies = [0.7151084082496034, 0.7059148598625066, 0.6774603384452671, 0.685027763088313, 0.6709148598625067, 0.6844368059227922, 0.6551084082496034, 0.6760364886303544, 0.6536567953463776, 0.6942345319936541, 0.6686700158646219, 0.7160629296668429, 0.727313590692755, 0.6984294024325753, 0.6240613432046536]
            f1_scores = [0.6757352083064074, 0.6601549996231629, 0.6460100304738876, 0.6455373537315283, 0.6427902615568601, 0.6490014642193461, 0.636370979087429, 0.6401367368205247, 0.6208250657981631, 0.6740247468326171, 0.6387443844786758, 0.6793595551605691, 0.6627108942715793, 0.5909533907874503, 0.5581566637372981]
        else:
            #threshold=0.7
            bal_accuracies = [0.6298640937856623, 0.6371511567589998, 0.6257044299201162, 0.6160665006743438, 0.653215582529308, 0.6577373171490818, 0.6662309368191721, 0.6561764705882352, 0.6794179894179895, 0.6891399522772073, 0.6759082892416226, 0.6858501919286233, 0.6430703392468098, 0.5541575889615106, 0.4164638447971781]
            f1_scores = [0.6280046751239983, 0.6363416268032769, 0.6184039506127835, 0.6111592707960752, 0.6483716508818267, 0.6514388545680916, 0.6608556263538345, 0.6464856533745816, 0.6593730406262736, 0.6529834225893891, 0.6507209051499754, 0.6634408758348541, 0.5979142971550508, 0.5205361667258581, 0.40492893714978273]
    elif MODEL == "KNN":
        if BINARY:
            #bal_accuracies = [0.6561964569011105, 0.6978503437334743, 0.6803093601269169, 0.666237440507668, 0.662809360126917, 0.692809360126917, 0.6713577472236911, 0.6629706504494977, 0.6421509783183501, 0.6789119513484929, 0.692809360126917, 0.6052564780539397, 0.6060483870967742, 0.6411832363828662, 0.6528225806451613, 0.6410483870967743, 0.6180116340560551, 0.6020967741935485, 0.5939925965097832, 0.5055248545742994]
            #f1_scores = [0.6749596766599707, 0.7266324572696569, 0.7095499922587016, 0.6910666105171919, 0.678310321738686, 0.700600705436316, 0.673880852705851, 0.6782346676246954, 0.6552141732133807, 0.6784614238654953, 0.7014526575731355, 0.6197191543796382, 0.6081485496632457, 0.6601014104215952, 0.6745093943562808, 0.6774419726967269, 0.6148149202563519, 0.6131511451528715, 0.5953543308783561, 0.5002157775892933]
            #threshold=0.8
            #bal_accuracies = [0.6120029085140137, 0.6095571126388154, 0.6271245372818616, 0.6927154944473823, 0.7276480698043363, 0.6461158117398201, 0.6617609730301427, 0.6911567953463775, 0.62465097831835, 0.614825489159175, 0.6121641988365945, 0.5911158117398203, 0.6112903225806452, 0.587325489159175, 0.5939925965097832, 0.5055248545742994]
            #f1_scores = [0.6357856783805304, 0.6338626062523234, 0.6479963517434314, 0.729734133691835, 0.7703491459736886, 0.6658116221950557, 0.7123231611673541, 0.7115639284698518, 0.631716974249678, 0.6145139073601548, 0.5981006662687886, 0.6033438376349806, 0.6196527130565589, 0.575317890397116, 0.5953543308783561, 0.5002157775892933]
            #threshold=0.7
            bal_accuracies = [0.6486964569011107, 0.6277287149656267, 0.6685893707033317, 0.6268416181914331, 0.6552154944473824, 0.6127154944473823, 0.5887638815441566, 0.5971641988365943, 0.614825489159175, 0.6121641988365945, 0.5911158117398203, 0.6112903225806452, 0.587325489159175, 0.5939925965097832, 0.5055248545742994]
            f1_scores = [0.6530290757182111, 0.656199515503876, 0.7028510237127616, 0.6603855611798526, 0.6884595709000811, 0.6412095983465775, 0.6059427691585402, 0.6146909743640915, 0.6145139073601548, 0.5981006662687886, 0.6033438376349806, 0.6196527130565589, 0.575317890397116, 0.5953543308783561, 0.5002157775892933]
        else:
            #bal_accuracies = [0.6298428260192965, 0.6316313932980598, 0.6327819275858492, 0.6446223674655046, 0.6396223674655046, 0.6239578794480755, 0.6126595082477435, 0.6056364768129474, 0.5837960369332917, 0.5912444236954041, 0.5844179894179894, 0.576923954767092, 0.5640854860462705, 0.5835615727772592, 0.6172232596742401, 0.5888800705467372, 0.5747883597883597, 0.5403501400560223, 0.5246540097520489, 0.45252412075941495]
            #f1_scores = [0.6470771582117573, 0.6467106972192276, 0.6497421722119074, 0.657095485279938, 0.6475179840951332, 0.6357037437470318, 0.6197357938747272, 0.6220927580168618, 0.5973138620979269, 0.5996799600082852, 0.6000477416193811, 0.5867279530828184, 0.5733109637248381, 0.5939561814948349, 0.6264527752834753, 0.5994320555057518, 0.582035039091393, 0.5537012151364973, 0.5241724036250079, 0.4360640023148575]
            #threshold=0.8
            #bal_accuracies = [0.6341887125220459, 0.6140860047722791, 0.6373155929038281, 0.6261012553169415, 0.6064934121796866, 0.6119255109451187, 0.5966272434899885, 0.6040626621018778, 0.6037540201265692, 0.6257812013694366, 0.5867927170868347, 0.5322388214545077, 0.5357962444236953, 0.5403501400560223, 0.5246540097520489, 0.45252412075941495]
            #f1_scores = [0.6512101693922745, 0.6274013086858568, 0.655510808200248, 0.6438993436370963, 0.6224544121391637, 0.6230628283216058, 0.612017928481492, 0.6207967120738196, 0.6181919668811318, 0.6437507901822619, 0.6021585396226085, 0.538073957754336, 0.5421032679987473, 0.5537012151364973, 0.5241724036250079, 0.4360640023148575]
            #threshold=0.7
            bal_accuracies = [0.6069135802469136, 0.6153911194107271, 0.6063995227720718, 0.6413352007469655, 0.6446104367672995, 0.6276014109347442, 0.6687021475256769, 0.6352251270878722, 0.6158667911609087, 0.5791020852785558, 0.5695544143583359, 0.5357962444236953, 0.5403501400560223, 0.5246540097520489, 0.45252412075941495]
            f1_scores = [0.6218396788353739, 0.6349465379181977, 0.6274629406649119, 0.6619347334733625, 0.6658002693853369, 0.6504960872196011, 0.6839338384046447, 0.6465410199909951, 0.6301217988602599, 0.5857100909537324, 0.5738442834559144, 0.5421032679987473, 0.5537012151364973, 0.5241724036250079, 0.4360640023148575]
    '''
    
    bal_accuracies = bal_accuracies[::-1]
    y = np.array(bal_accuracies)
    f1_scores = f1_scores[::-1]
    y_ = np.array(f1_scores)
    x = np.array(range(1, len(y)+1))
        
    elbow_point_idx = find_elbow_point(x, y)
    elbow_point_num = elbow_point_idx + 1
    elbow_point_acc = y[elbow_point_idx]
    corr_f1 = y_[elbow_point_idx]
    #print(f"The elbow point for bal.accuracy is at {elbow_point_num} features.")
    #print(f"Bal.accuracy at the elbow point: {elbow_point_acc:.2f}")
    #print(f"Corresponding F1-score: {corr_f1:.2f}")
        
    elbow_point_f1_idx = find_elbow_point(x, y_)
    elbow_point_f1_num = elbow_point_f1_idx + 1
    elbow_point_f1 = y_[elbow_point_f1_idx]
    corr_acc = y[elbow_point_f1_idx]
    print(f"The elbow point for F1-score is at {elbow_point_f1_num} features.")
    print(f"F1-score at the elbow point: {elbow_point_f1:.2f}")
    #print(f"Corresponding bal.accuracy: {corr_acc:.2f}")
        
    filename = "chs"
    if BINARY:
        filename = filename + "_binary_"
    else:
        filename = filename + "_3classes_"
    filename = filename + MODEL
            
    # Plot accuracies
    fig0, ax0 = plt.subplots()
    ax0.plot(x, y, marker='o')
            
    ax0.set_xlabel('Number of Features', fontsize=14)
    ax0.set_ylabel('Balanced accuracy', fontsize=14)
    ax0.tick_params(axis='both', which='major', labelsize=12)
    ax0.tick_params(axis='both', which='minor', labelsize=10)
    plt.grid(True)
    acc_filename = filename + "_bal_acc.png"
    full_filename = os.path.join(FIG_DIR, acc_filename)
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(full_filename, dpi=600)
    plt.show()
            
    # Plot F1-scores
    fig2, ax2 = plt.subplots()
    ax2.plot(x, y_, marker='o')
    ax2.set_xlabel('Number of features', fontsize=14)
    ax2.set_ylabel('f1 score', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='minor', labelsize=10)
    plt.grid(True)
    f1_filename = filename + "_f1.png"
    full_filename = os.path.join(FIG_DIR, f1_filename)
    #plt.tight_layout()
    plt.savefig(full_filename, dpi=600)
    plt.show()
    
    
        
    max_acc = max(bal_accuracies)
    max_index = bal_accuracies.index(max_acc)
    #print(f"Max_index: {max_index}")
    number_of_features = max_index + 1
    acc = bal_accuracies[max_index]
    f1 = f1_scores[max_index]
    #print("Optimal number of features by maximizing Bal.Accuracy")
    #print(f"Optimal number of features: {number_of_features}, Bal.Accuracy: {acc:.2f}, F1-score: {f1:.2f}")

    max_f1 = max(f1_scores)
    max_index = f1_scores.index(max_f1)
    #print(f"Max_index: {max_index}")
    number_of_features = max_index + 1
    acc = bal_accuracies[max_index]
    f1 = f1_scores[max_index]
    print("Optimal number of features by maximizing F1-score")
    #print(f"Optimal number of features: {number_of_features}, Bal.Accuracy: {acc:.2f}, F1-score: {f1:.2f}")
    print(f"Optimal number of features: {number_of_features}, F1-score: {f1:.2f}")
    
    if SAVE_COORD:
       
        filename = 'curve_coordinates.csv'
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['x', 'y'])  # Write the header
            for i in range(len(x)):
                writer.writerow([x[i], y_[i]])
##############        

start_time = time.time()

main()

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.3f} seconds")
    