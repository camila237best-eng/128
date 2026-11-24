// puzzle.cpp - Recherche de clés privées Bitcoin (secp256k1) multi-cibles, multi-cœurs, sans dépendance.
//
// Compilation : g++ -O3 -march=native -std=c++17 -o puzzle puzzle.cpp   (Linux : ajouter -pthread)
//               (-O3 et -march=native sont importants : sans eux le programme est 2 à 3 fois plus lent)
//   Windows/MinGW : si le programme s'arrête brutalement, ajoute -mno-avx à la compilation.
// ~ $ ./lesly 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU 71 address -v10

// Usage :
//   puzzle <clé_publique | adresse | cibles.txt | cibles.bloom> <bits | début:fin> [address] [mode] [options]
//   puzzle --info                                   affiche les cœurs utilisables
//   puzzle --selftest                               lance uniquement les auto-tests
//   puzzle --genpub <clé_privée_hex> | --genaddr <clé_privée_hex>     (utilitaires de test)
//
// La clé recherchée est dans [2^(bits-1) , 2^bits - 1], ou dans la plage début:fin donnée en hexadécimal
// (n'importe quelle plage, ex. 2000000:3ffffff).
// address      : à ajouter quand les cibles sont des adresses Bitcoin (1... ou bc1q...)
// cibles.txt   : une cible par ligne -> construit et sauvegarde cibles.bloom ; cibles.bloom : chargement direct
// Mode (un seul par lancement ; tous les cœurs travaillent ensemble sur la même suite de blocs) :
//   -r aléatoire (défaut)   -s séquentiel   -sobol   -pi   -e   -ln   -m Monte Carlo
// -vN : permutation des N derniers bits (ex. -v10 ; -v 10 accepté). Chaque clé générée par la suite (mode et nombre de bits choisis,
//        ou -pNN) est suivie du test de TOUTES les valeurs de ses N bits faibles (2^N clés alignées, mêmes bits hauts) : si le
//        préfixe est le bon, la clé est trouvée. Blocs disjoints (aucune clé testée deux fois). N de 1 à 62 ; N >= 10 conseillé.
//        Au-delà de 2^26 blocs, la suite est calculée sans mémoire : -s et -sobol restent exhaustifs, les autres tirent sans exclusion.
// -t N : nombre de cœurs (par défaut : tous les cœurs utilisables, quotas de conteneur compris)
// Sorties / reprise :
//   --out DIR       dossier des résultats et du point de reprise
//   --ckpt FILE     fichier de reprise (défaut : DIR/puzzle_<empreinte>.ckpt) ; --no-ckpt le désactive ; --fresh l'ignore
//   --ckpt-every S  intervalle de sauvegarde en secondes (défaut 120)
//   --time T        arrêt propre après T (ex. 3600, 90m, 11h30m) : le point de reprise est écrit ; relancer la même commande reprend
//   --status S      intervalle (s) de la ligne de récapitulatif (défaut 60)
//   --skip-test     saute les auto-tests au démarrage (utile pour une reprise rapide)
// Mode vanity (adresses Bitcoin personnalisées, remplace <cible> <bits> ; les modes ci-dessus ne sont pas modifiés) :
//   puzzle --vanity <préfixe> [plage] [-part i:n] [-s|-sobol|-pi|-e|-ln|-m] [-t N] [sorties / reprise]
//   --vanity P    adresses P2PKH "1..." (clé compressée), préfixe Base58 sensible à la casse ("1" initial facultatif).
//                 Résultat : adresse + clé hex + WIF dans kangaroo_vanity.txt (dossier --out). Probabilité exacte par clé affichée.
//   plage         -b 70 -> [2^69, 2^70-1] | -bb 40 -> [1, 2^40-1] | -r début:fin (hexadécimal) | -r début -b N -> début + 2^N - 1
//                 (sans plage : tout l'espace des clés, ordre aléatoire). Exploration exhaustive sans chevauchement ;
//                 la recherche continue après chaque résultat jusqu'à "plage entièrement explorée" (ou Ctrl-C / --time).
//   -part i:n     découpe la plage en n parties (taille arrondie au-dessus, sans trou ni chevauchement), explore la partie i (1..n)
// (compiler avec -std=c++17 : <filesystem> est utilisé pour créer le dossier --out)
// Vanity : -vanity et --vanity sont équivalents ; ordre des blocs ALÉATOIRE par défaut (-s : séquentiel).
// Affichage (tous les modes) : pas de tableau. Chaque clé trouvée est annoncée tout de suite (et enregistrée), puis une ligne de
//   récapitulatif (progression, vitesse, écoulé, reste, trouvées) est ajoutée toutes les 60 s (--status S pour changer).
#define _FILE_OFFSET_BITS 64
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <csignal>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <atomic>
#include <exception>
#include <ctime>
#include <set>
#include <map>
#include <filesystem>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <thread>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#endif
#endif

typedef uint64_t u64;
typedef uint32_t u32;
typedef unsigned __int128 u128;

struct U256 { u64 d[4]; };               // little-endian (d[0] = poids faible)
static const U256 ZERO = {{0,0,0,0}};
static const U256 ONE  = {{1,0,0,0}};
static const U256 PRIME = {{0xFFFFFFFEFFFFFC2FULL, ~0ULL, ~0ULL, ~0ULL}};
static const u64  KC = 0x1000003D1ULL;   // 2^256 mod p

// ---------------------------------------------------------------- 256 bits brut
static inline bool is_zero(const U256& a){ return (a.d[0]|a.d[1]|a.d[2]|a.d[3])==0; }
static inline bool eq(const U256& a,const U256& b){
    return ((a.d[0]^b.d[0])|(a.d[1]^b.d[1])|(a.d[2]^b.d[2])|(a.d[3]^b.d[3]))==0;
}
static inline bool geq(const U256& a,const U256& b){
    for(int i=3;i>=0;--i){ if(a.d[i]!=b.d[i]) return a.d[i]>b.d[i]; }
    return true;
}
static inline u64 add_raw(U256& r,const U256& a,const U256& b){
    u128 acc=0; u64 c=0;
    for(int i=0;i<4;i++){ acc=(u128)a.d[i]+b.d[i]+c; r.d[i]=(u64)acc; c=(u64)(acc>>64); }
    return c;
}
static inline u64 sub_raw(U256& r,const U256& a,const U256& b){
    u64 br=0;
    for(int i=0;i<4;i++){
        u128 t=(u128)a.d[i]-b.d[i]-br;
        r.d[i]=(u64)t; br=(u64)((t>>64)&1);
    }
    return br;
}
static U256 div_small(U256 a,u64 d){                       // a / d (d > 0)
    u128 rem=0;
    for(int i=3;i>=0;--i){ u128 cur=(rem<<64)|a.d[i]; a.d[i]=(u64)(cur/d); rem=cur%d; }
    return a;
}
static U256 shr_n(U256 a,int n){
    for(int s2=0;s2<n;s2++){ for(int k=0;k<4;k++){ a.d[k]>>=1; if(k<3) a.d[k]|=a.d[k+1]<<63; } }
    return a;
}
static inline void add_small(U256& a,u64 v){
    u128 acc=(u128)a.d[0]+v; a.d[0]=(u64)acc; u64 c=(u64)(acc>>64);
    for(int i=1;i<4;i++){ acc=(u128)a.d[i]+c; a.d[i]=(u64)acc; c=(u64)(acc>>64); }
}
static inline int get_bit(const U256& a,int i){ return (int)((a.d[i>>6]>>(i&63))&1); }
static int highest_bit(const U256& a){
    for(int i=255;i>=0;--i) if(get_bit(a,i)) return i;
    return -1;
}
static U256 shr2(U256 a){
    for(int i=0;i<4;i++){
        a.d[i]>>=2;
        if(i<3) a.d[i]|=a.d[i+1]<<62;   // a.d[i+1] n'a pas encore été décalé
    }
    return a;
}

// ---------------------------------------------------------------- hex / décimal
static bool parse_hex(const std::string& s,U256& out){
    if(s.empty()||s.size()>64) return false;
    out=ZERO;
    for(size_t i=0;i<s.size();i++){
        char c=s[s.size()-1-i]; int v;
        if(c>='0'&&c<='9') v=c-'0'; else if(c>='a'&&c<='f') v=c-'a'+10;
        else if(c>='A'&&c<='F') v=c-'A'+10; else return false;
        out.d[i/16]|=(u64)v<<((i%16)*4);
    }
    return true;
}
static std::string to_hex(const U256& a){
    char buf[65];
    snprintf(buf,sizeof buf,"%016llx%016llx%016llx%016llx",
        (unsigned long long)a.d[3],(unsigned long long)a.d[2],
        (unsigned long long)a.d[1],(unsigned long long)a.d[0]);
    return buf;
}
static std::string to_dec(U256 a){
    if(is_zero(a)) return "0";
    std::vector<u64> parts;
    while(!is_zero(a)){
        u128 rem=0;
        for(int i=3;i>=0;--i){
            u128 cur=(rem<<64)|a.d[i];
            a.d[i]=(u64)(cur/1000000000ULL);
            rem=cur%1000000000ULL;
        }
        parts.push_back((u64)rem);
    }
    char buf[32]; std::string s;
    snprintf(buf,sizeof buf,"%llu",(unsigned long long)parts.back()); s+=buf;
    for(int i=(int)parts.size()-2;i>=0;--i){ snprintf(buf,sizeof buf,"%09llu",(unsigned long long)parts[i]); s+=buf; }
    return s;
}

// ---------------------------------------------------------------- corps Fp (p = 2^256 - 2^32 - 977)
static inline void fe_add(U256& r,const U256& a,const U256& b){
    U256 t; u64 c=add_raw(t,a,b);
    if(c){ U256 k={{KC,0,0,0}}; add_raw(t,t,k); }
    if(geq(t,PRIME)) sub_raw(t,t,PRIME);
    r=t;
}
static inline void fe_sub(U256& r,const U256& a,const U256& b){
    U256 t; u64 br=sub_raw(t,a,b);
    if(br){ U256 k={{KC,0,0,0}}; sub_raw(t,t,k); }
    r=t;
}
// Réduction d'un produit de 512 bits modulo p (2^256 = 2^32 + 977 mod p)
static inline void fe_reduce(U256& r,const u64* t){
    u64 r5[5]; u64 carry=0;
    for(int i=0;i<4;i++){
        u128 acc=(u128)t[4+i]*KC+t[i]+carry;
        r5[i]=(u64)acc; carry=(u64)(acc>>64);
    }
    r5[4]=carry;
    U256 res; u128 acc=(u128)r5[4]*KC+r5[0];
    res.d[0]=(u64)acc; carry=(u64)(acc>>64);
    for(int i=1;i<4;i++){ acc=(u128)r5[i]+carry; res.d[i]=(u64)acc; carry=(u64)(acc>>64); }
    if(carry){
        acc=(u128)res.d[0]+KC; res.d[0]=(u64)acc; carry=(u64)(acc>>64);
        for(int i=1;i<4;i++){ acc=(u128)res.d[i]+carry; res.d[i]=(u64)acc; carry=(u64)(acc>>64); }
    }
    if(geq(res,PRIME)) sub_raw(res,res,PRIME);
    r=res;
}
static inline void fe_mul(U256& r,const U256& a,const U256& b){
    u64 t[8]={0,0,0,0,0,0,0,0};
    for(int i=0;i<4;i++){
        u64 carry=0;
        for(int j=0;j<4;j++){
            u128 acc=(u128)a.d[i]*b.d[j]+t[i+j]+carry;
            t[i+j]=(u64)acc; carry=(u64)(acc>>64);
        }
        t[i+4]=carry;
    }
    fe_reduce(r,t);
}
// Carré dédié : 10 produits de 64 bits au lieu de 16
static inline void fe_sqr(U256& r,const U256& a){
    u64 t[8]={0,0,0,0,0,0,0,0};
    for(int i=0;i<3;i++){
        u64 carry=0;
        for(int j=i+1;j<4;j++){
            u128 acc=(u128)a.d[i]*a.d[j]+t[i+j]+carry;
            t[i+j]=(u64)acc; carry=(u64)(acc>>64);
        }
        t[i+4]=carry;
    }
    u64 c=0;
    for(int k=0;k<8;k++){ u64 v=t[k]; t[k]=(v<<1)|c; c=v>>63; }
    u64 carry=0;
    for(int i=0;i<4;i++){
        u128 acc=(u128)a.d[i]*a.d[i]+t[2*i]+carry;
        t[2*i]=(u64)acc; carry=(u64)(acc>>64);
        acc=(u128)t[2*i+1]+carry;
        t[2*i+1]=(u64)acc; carry=(u64)(acc>>64);
    }
    fe_reduce(r,t);
}
static inline void fe_sqr_n(U256& r,const U256& a,int n){
    U256 t=a; for(int i=0;i<n;i++) fe_sqr(t,t); r=t;
}
static U256 fe_pow(const U256& base,const U256& e){
    U256 r=ONE; int top=highest_bit(e);
    for(int i=top;i>=0;--i){ fe_sqr(r,r); if(get_bit(e,i)) fe_mul(r,r,base); }
    return r;
}
// Inverse modulaire a^(p-2) par chaîne d'additions : 255 carrés + 15 produits
static U256 fe_inv(const U256& a){
    U256 x2,x3,x6,x9,x11,x22,x44,x88,x176,x220,x223,t;
    fe_sqr(x2,a); fe_mul(x2,x2,a);
    fe_sqr(x3,x2); fe_mul(x3,x3,a);
    fe_sqr_n(x6,x3,3); fe_mul(x6,x6,x3);
    fe_sqr_n(x9,x6,3); fe_mul(x9,x9,x3);
    fe_sqr_n(x11,x9,2); fe_mul(x11,x11,x2);
    fe_sqr_n(x22,x11,11); fe_mul(x22,x22,x11);
    fe_sqr_n(x44,x22,22); fe_mul(x44,x44,x22);
    fe_sqr_n(x88,x44,44); fe_mul(x88,x88,x44);
    fe_sqr_n(x176,x88,88); fe_mul(x176,x176,x88);
    fe_sqr_n(x220,x176,44); fe_mul(x220,x220,x44);
    fe_sqr_n(x223,x220,3); fe_mul(x223,x223,x3);
    fe_sqr_n(t,x223,23); fe_mul(t,t,x22);
    fe_sqr_n(t,t,5); fe_mul(t,t,a);
    fe_sqr_n(t,t,3); fe_mul(t,t,x2);
    fe_sqr_n(t,t,2); fe_mul(t,t,a);
    return t;
}

// ---------------------------------------------------------------- courbe y^2 = x^3 + 7
struct Aff { U256 x,y; bool inf; };
struct Jac { U256 X,Y,Z; bool inf; };

static Aff GEN;
static const char* GX_HEX="79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
static const char* GY_HEX="483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8";

static bool on_curve(const Aff& P){
    U256 l,r; fe_sqr(l,P.y); fe_sqr(r,P.x); fe_mul(r,r,P.x);
    U256 seven={{7,0,0,0}}; fe_add(r,r,seven);
    return eq(l,r);
}

// --- Arithmétique affine (lente, 1 inversion par opération) : sert à la vérification indépendante
static Aff aff_add(const Aff& P,const Aff& Q){
    if(P.inf) return Q;
    if(Q.inf) return P;
    U256 l;
    if(eq(P.x,Q.x)){
        if(!eq(P.y,Q.y) || is_zero(P.y)){ Aff o; o.x=ZERO;o.y=ZERO;o.inf=true; return o; }
        U256 n,d; fe_sqr(n,P.x); U256 n3; fe_add(n3,n,n); fe_add(n3,n3,n);   // 3x^2
        fe_add(d,P.y,P.y);
        fe_mul(l,n3,fe_inv(d));
    } else {
        U256 n,d; fe_sub(n,Q.y,P.y); fe_sub(d,Q.x,P.x);
        fe_mul(l,n,fe_inv(d));
    }
    Aff R; R.inf=false;
    fe_sqr(R.x,l); fe_sub(R.x,R.x,P.x); fe_sub(R.x,R.x,Q.x);
    U256 t; fe_sub(t,P.x,R.x); fe_mul(R.y,l,t); fe_sub(R.y,R.y,P.y);
    return R;
}
static Aff aff_mul(const U256& k,const Aff& P){
    Aff R; R.x=ZERO;R.y=ZERO;R.inf=true;
    for(int i=highest_bit(k);i>=0;--i){
        R=aff_add(R,R);
        if(get_bit(k,i)) R=aff_add(R,P);
    }
    return R;
}

// --- Arithmétique Jacobienne (rapide) : sert au calcul du point de départ
static Jac jac_dbl(const Jac& P){
    if(P.inf||is_zero(P.Y)){ Jac o; o.X=o.Y=o.Z=ZERO; o.inf=true; return o; }
    U256 A,B,C,D,E,F,t,t2;
    fe_sqr(A,P.X); fe_sqr(B,P.Y); fe_sqr(C,B);
    fe_add(t,P.X,B); fe_sqr(t,t); fe_sub(t,t,A); fe_sub(t,t,C); fe_add(D,t,t);
    fe_add(E,A,A); fe_add(E,E,A);
    fe_sqr(F,E);
    Jac R; R.inf=false;
    fe_add(t,D,D); fe_sub(R.X,F,t);
    fe_sub(t,D,R.X); fe_mul(t,E,t);
    fe_add(t2,C,C); fe_add(t2,t2,t2); fe_add(t2,t2,t2);   // 8C
    fe_sub(R.Y,t,t2);
    fe_mul(t,P.Y,P.Z); fe_add(R.Z,t,t);
    return R;
}
static Jac jac_madd(const Jac& P,const Aff& Q){
    if(P.inf){ Jac R; R.X=Q.x;R.Y=Q.y;R.Z=ONE;R.inf=false; return R; }
    U256 Z1Z1,U2,S2,H,HH,I,J,r,V,t;
    fe_sqr(Z1Z1,P.Z); fe_mul(U2,Q.x,Z1Z1);
    fe_mul(S2,Q.y,P.Z); fe_mul(S2,S2,Z1Z1);
    fe_sub(H,U2,P.X);
    fe_sub(t,S2,P.Y); fe_add(r,t,t);
    if(is_zero(H)){
        if(is_zero(r)) return jac_dbl(P);
        Jac o; o.X=o.Y=o.Z=ZERO; o.inf=true; return o;
    }
    fe_sqr(HH,H); fe_add(I,HH,HH); fe_add(I,I,I);
    fe_mul(J,H,I); fe_mul(V,P.X,I);
    Jac R; R.inf=false;
    fe_sqr(R.X,r); fe_sub(R.X,R.X,J); fe_sub(R.X,R.X,V); fe_sub(R.X,R.X,V);
    fe_sub(t,V,R.X); fe_mul(t,r,t);
    U256 u; fe_mul(u,P.Y,J); fe_add(u,u,u);
    fe_sub(R.Y,t,u);
    fe_add(t,P.Z,H); fe_sqr(t,t); fe_sub(t,t,Z1Z1); fe_sub(R.Z,t,HH);
    return R;
}
static Jac jac_mul(const U256& k,const Aff& P){
    Jac R; R.X=R.Y=R.Z=ZERO; R.inf=true;
    for(int i=highest_bit(k);i>=0;--i){
        R=jac_dbl(R);
        if(get_bit(k,i)) R=jac_madd(R,P);
    }
    return R;
}
static Aff jac_to_aff(const Jac& J){
    Aff A; A.x=ZERO;A.y=ZERO;
    if(J.inf){ A.inf=true; return A; }
    U256 zi=fe_inv(J.Z),zi2,zi3;
    fe_sqr(zi2,zi); fe_mul(zi3,zi2,zi);
    fe_mul(A.x,J.X,zi2); fe_mul(A.y,J.Y,zi3); A.inf=false;
    return A;
}

// ---------------------------------------------------------------- utilitaires divers
static std::string compress_hex(const Aff& P){
    return std::string(((P.y.d[0]&1)?"03":"02"))+to_hex(P.x);
}
static inline u64 splitmix(u64& s){
    u64 z=(s+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL;
    return z^(z>>31);
}
struct Rng {
    u64 s[4];
    void seed(u64 v){ for(int i=0;i<4;i++) s[i]=splitmix(v); }
    static inline u64 rotl(u64 x,int k){ return (x<<k)|(x>>(64-k)); }
    u64 next(){
        u64 r=rotl(s[1]*5,7)*9, t=s[1]<<17;
        s[2]^=s[0]; s[3]^=s[1]; s[1]^=s[2]; s[0]^=s[3]; s[2]^=t; s[3]=rotl(s[3],45);
        return r;
    }
};
struct SpinLock {
    std::atomic_flag f=ATOMIC_FLAG_INIT;
    void lock(){ while(f.test_and_set(std::memory_order_acquire)){} }
    void unlock(){ f.clear(std::memory_order_release); }
};
static U256 shl_u64(u64 v,int sh){
    U256 r=ZERO; int limb=sh>>6, off=sh&63;
    if(limb<4) r.d[limb]|=v<<off;
    if(off && limb+1<4) r.d[limb+1]|=v>>(64-off);
    return r;
}
static U256 low_mask(int nbits){           // 2^nbits - 1
    U256 r=ZERO;
    for(int b=0;b<nbits;b++) r.d[b>>6]|=1ULL<<(b&63);
    return r;
}
static std::string fmt_int(u64 v){
    std::string s=std::to_string(v), o;
    int c=0;
    for(int i=(int)s.size()-1;i>=0;--i){ o.insert(o.begin(),s[i]); if(++c%3==0&&i>0) o.insert(o.begin(),' '); }
    return o;
}
// Affichage compact avec multiples : 847 / 1.23 K / 12.3 M / 456 G ... (3 chiffres significatifs)
static std::string fmt_si(long double v){
    const char* u[]={"","K","M","G","T","P","E"};
    int i=0; while(v>=999.5L && i<6){ v/=1000.0L; i++; }
    char b[32];
    if(!i) snprintf(b,sizeof b,"%.0Lf",v);
    else snprintf(b,sizeof b,v<9.995L?"%.2Lf %s":(v<99.95L?"%.1Lf %s":"%.0Lf %s"),v,u[i]);
    return b;
}
static std::string fmt_keys(long double v){ return fmt_si(v); }
// Vitesse (clés/s) avec multiple automatique : 850 k/s, 1.25 Mk/s, 3.10 Gk/s
static std::string fmt_rate(double keys_per_s){
    const char* u[]={"","K","M","G","T"};
    long double v=keys_per_s<0?0:keys_per_s; int i=0; while(v>=999.5L && i<4){ v/=1000.0L; i++; }
    char b[40]; snprintf(b,sizeof b,v<9.995L?"%.2Lf %sk/s":(v<99.95L?"%.1Lf %sk/s":"%.0Lf %sk/s"),v,u[i]); return b;
}
static std::string fmt_time(long double s){
    char b[64];
    if(s<0||s!=s) return "?";
    if(s>3.15e9L) return ">100 ans";
    unsigned long long t=(unsigned long long)s;
    unsigned long long y=t/31536000ULL; t%=31536000ULL;
    unsigned long long d=t/86400, h=(t%86400)/3600, m=(t%3600)/60, sec=t%60;
    if(y) snprintf(b,sizeof b,"%llu an(s) %llu j",y,d);
    else if(d) snprintf(b,sizeof b,"%llu j %02lluh %02llum",d,h,m);
    else snprintf(b,sizeof b,"%02llu:%02llu:%02llu",h,m,sec);
    return b;
}
static std::string short_hex(const U256& v){
    std::string h=to_hex(v); size_t p=h.find_first_not_of('0');
    h=(p==std::string::npos)?"0":h.substr(p);
    if(h.size()>22) h=h.substr(0,10)+"…"+h.substr(h.size()-10);
    return "0x"+h;
}

// ---------------------------------------------------------------- SHA-256 / RIPEMD-160 (un seul bloc) et adresses
static inline u32 rotr32(u32 x,int n){ return (x>>n)|(x<<(32-n)); }
static inline u32 rotl32(u32 x,int n){ return (x<<n)|(x>>(32-n)); }
static const u32 SHA_K[64]={
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

// SHA-256 d'un message tenant dans un seul bloc (w = 16 mots big-endian déjà préparés)
static inline void sha256_block(const u32* w16,u32* out){
    u32 w[64];
    for(int i=0;i<16;i++) w[i]=w16[i];
    #pragma GCC unroll 64
    for(int i=16;i<64;i++){
        u32 s0=rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);
        u32 s1=rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    u32 a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a,e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,h=0x5be0cd19;
    #pragma GCC unroll 64
    for(int i=0;i<64;i++){
        u32 S1=rotr32(e,6)^rotr32(e,11)^rotr32(e,25), ch=(e&f)^(~e&g);
        u32 t1=h+S1+ch+SHA_K[i]+w[i];
        u32 S0=rotr32(a,2)^rotr32(a,13)^rotr32(a,22), mj=(a&b)^(a&c)^(b&c);
        u32 t2=S0+mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    out[0]=a+0x6a09e667; out[1]=b+0xbb67ae85; out[2]=c+0x3c6ef372; out[3]=d+0xa54ff53a;
    out[4]=e+0x510e527f; out[5]=f+0x9b05688c; out[6]=g+0x1f83d9ab; out[7]=h+0x5be0cd19;
}
// SHA-256 d'un message de n <= 55 octets (octet par octet)
static void sha256_bytes(const unsigned char* m,size_t n,unsigned char* out32){
    unsigned char buf[64]; memset(buf,0,64); memcpy(buf,m,n); buf[n]=0x80;
    u64 bl=(u64)n*8; for(int i=0;i<8;i++) buf[63-i]=(unsigned char)(bl>>(8*i));
    u32 w[16],o[8];
    for(int i=0;i<16;i++) w[i]=((u32)buf[4*i]<<24)|((u32)buf[4*i+1]<<16)|((u32)buf[4*i+2]<<8)|buf[4*i+3];
    sha256_block(w,o);
    for(int i=0;i<8;i++){ out32[4*i]=o[i]>>24; out32[4*i+1]=o[i]>>16; out32[4*i+2]=o[i]>>8; out32[4*i+3]=o[i]; }
}

static const unsigned char RMD_R1[80]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2, 4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
static const unsigned char RMD_R2[80]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14, 12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
static const unsigned char RMD_S1[80]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12, 9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
static const unsigned char RMD_S2[80]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8, 8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
static const u32 RMD_K1[5]={0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
static const u32 RMD_K2[5]={0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};
template<int F> static inline u32 rmd_f(u32 x,u32 y,u32 z){
    if(F==0) return x^y^z;
    if(F==1) return (x&y)|(~x&z);
    if(F==2) return (x|~y)^z;
    if(F==3) return (x&z)|(y&~z);
    return x^(y|~z);
}
// RIPEMD-160 d'un message tenant dans un seul bloc (X = 16 mots little-endian)
template<int F> static inline void rmd_seg(const u32* X,u32& a,u32& b,u32& c,u32& d,u32& e,
                                           u32& a2,u32& b2,u32& c2,u32& d2,u32& e2){
    #pragma GCC unroll 16
    for(int j=16*F;j<16*F+16;j++){
        u32 t=rotl32(a+rmd_f<F>(b,c,d)+X[RMD_R1[j]]+RMD_K1[F],RMD_S1[j])+e;
        a=e; e=d; d=rotl32(c,10); c=b; b=t;
        t=rotl32(a2+rmd_f<4-F>(b2,c2,d2)+X[RMD_R2[j]]+RMD_K2[F],RMD_S2[j])+e2;
        a2=e2; e2=d2; d2=rotl32(c2,10); c2=b2; b2=t;
    }
}
static inline void rmd160_block(const u32* X,u32* out){
    u32 a=0x67452301,b=0xEFCDAB89,c=0x98BADCFE,d=0x10325476,e=0xC3D2E1F0;
    u32 a2=a,b2=b,c2=c,d2=d,e2=e;
    rmd_seg<0>(X,a,b,c,d,e,a2,b2,c2,d2,e2);
    rmd_seg<1>(X,a,b,c,d,e,a2,b2,c2,d2,e2);
    rmd_seg<2>(X,a,b,c,d,e,a2,b2,c2,d2,e2);
    rmd_seg<3>(X,a,b,c,d,e,a2,b2,c2,d2,e2);
    rmd_seg<4>(X,a,b,c,d,e,a2,b2,c2,d2,e2);
    const u32 I0=0x67452301u, I1=0xEFCDAB89u, I2=0x98BADCFEu, I3=0x10325476u, I4=0xC3D2E1F0u;
    u32 t=I1+c+d2;
    out[1]=I2+d+e2; out[2]=I3+e+a2; out[3]=I4+a+b2; out[4]=I0+b+c2; out[0]=t;
}
// hash160 empaqueté dans un U256 : lecture little-endian des 20 octets du condensat
static inline U256 pack_rmd(const u32* h){
    U256 r; r.d[0]=h[0]|((u64)h[1]<<32); r.d[1]=h[2]|((u64)h[3]<<32); r.d[2]=h[4]; r.d[3]=0; return r;
}
static U256 pack20(const unsigned char* b){
    U256 r=ZERO; memcpy(&r.d[0],b,8); memcpy(&r.d[1],b+8,8); u32 t; memcpy(&t,b+16,4); r.d[2]=t; return r;
}
static void unpack20(const U256& h,unsigned char* b){
    memcpy(b,&h.d[0],8); memcpy(b+8,&h.d[1],8); u32 t=(u32)h.d[2]; memcpy(b+16,&t,4);
}
// Chemin rapide (boucle de recherche) : hash160 de la clé publique compressée (parité de Y, X)
static inline U256 hash160_fast(const U256& x,u64 par){
    u32 X[8];
    X[0]=(u32)(x.d[3]>>32); X[1]=(u32)x.d[3]; X[2]=(u32)(x.d[2]>>32); X[3]=(u32)x.d[2];
    X[4]=(u32)(x.d[1]>>32); X[5]=(u32)x.d[1]; X[6]=(u32)(x.d[0]>>32); X[7]=(u32)x.d[0];
    u32 w[16],H[8],rw[16],o[5];
    w[0]=((0x02u+(u32)par)<<24)|(X[0]>>8);
    for(int i=1;i<8;i++) w[i]=(X[i-1]<<24)|(X[i]>>8);
    w[8]=(X[7]<<24)|0x00800000u;
    for(int i=9;i<15;i++) w[i]=0;
    w[15]=264;
    sha256_block(w,H);
    for(int i=0;i<8;i++) rw[i]=__builtin_bswap32(H[i]);
    rw[8]=0x80; for(int i=9;i<14;i++) rw[i]=0; rw[14]=256; rw[15]=0;
    rmd160_block(rw,o);
    return pack_rmd(o);
}
// Chemin "octets" (vérifications indépendantes)
static void pub33(const U256& x,u64 par,unsigned char* out){
    out[0]=(unsigned char)(2+par);
    for(int i=0;i<32;i++) out[1+i]=(unsigned char)(x.d[3-i/8]>>(56-8*(i%8)));
}
static void hash160_bytes(const unsigned char* m,size_t n,unsigned char* out20){
    unsigned char dg[32]; sha256_bytes(m,n,dg);
    u32 X[16]={0},o[5];
    for(int i=0;i<8;i++) X[i]=(u32)dg[4*i]|((u32)dg[4*i+1]<<8)|((u32)dg[4*i+2]<<16)|((u32)dg[4*i+3]<<24);
    X[8]=0x80; X[14]=256;
    rmd160_block(X,o);
    for(int i=0;i<5;i++){ out20[4*i]=o[i]; out20[4*i+1]=o[i]>>8; out20[4*i+2]=o[i]>>16; out20[4*i+3]=o[i]>>24; }
}
static U256 hash160_point_bytes(const Aff& P){
    unsigned char pb[33],h[20]; pub33(P.x,P.y.d[0]&1,pb); hash160_bytes(pb,33,h); return pack20(h);
}

// ---- hash160 de plusieurs clés à la fois (vecteurs SIMD via les extensions GCC/Clang)
#if defined(VLANES)
#elif defined(_WIN32)
#define VLANES 4          // 128 bits : évite les soucis d'alignement de pile 32 octets de MinGW
#else
#define VLANES 8          // 256 bits (AVX2 si -march=native)
#endif
typedef u32 vu __attribute__((vector_size(4*VLANES)));
static inline vu vsplat(u32 x){ vu v; for(int i=0;i<VLANES;i++) v[i]=x; return v; }
#define VROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define VROTL(x,n) (((x)<<(n))|((x)>>(32-(n))))
template<int F> static inline vu rmd_fv(vu x,vu y,vu z){
    if(F==0) return x^y^z;
    if(F==1) return (x&y)|(~x&z);
    if(F==2) return (x|~y)^z;
    if(F==3) return (x&z)|(y&~z);
    return x^(y|~z);
}
template<int F> static inline void rmd_segv(const vu* X,vu& a,vu& b,vu& c,vu& d,vu& e,vu& a2,vu& b2,vu& c2,vu& d2,vu& e2){
    #pragma GCC unroll 16
    for(int j=16*F;j<16*F+16;j++){
        vu t=VROTL(a+rmd_fv<F>(b,c,d)+X[RMD_R1[j]]+vsplat(RMD_K1[F]),RMD_S1[j])+e;
        a=e; e=d; d=VROTL(c,10); c=b; b=t;
        t=VROTL(a2+rmd_fv<4-F>(b2,c2,d2)+X[RMD_R2[j]]+vsplat(RMD_K2[F]),RMD_S2[j])+e2;
        a2=e2; e2=d2; d2=VROTL(c2,10); c2=b2; b2=t;
    }
}
// hash160 de VLANES clés publiques compressées (xs[l], pars[l]) -> out[l] (même empaquetage que hash160_fast)
static void hash160_vec(const U256* xs,const u64* pars,U256* out){
    u32 Xl[8][VLANES], Pl[VLANES];
    for(int l=0;l<VLANES;l++){
        const U256& x=xs[l];
        Xl[0][l]=(u32)(x.d[3]>>32); Xl[1][l]=(u32)x.d[3]; Xl[2][l]=(u32)(x.d[2]>>32); Xl[3][l]=(u32)x.d[2];
        Xl[4][l]=(u32)(x.d[1]>>32); Xl[5][l]=(u32)x.d[1]; Xl[6][l]=(u32)(x.d[0]>>32); Xl[7][l]=(u32)x.d[0];
        Pl[l]=0x02u+(u32)pars[l];
    }
    vu X[8],w[64],H[8],rw[16];
    for(int i=0;i<8;i++) memcpy(&X[i],Xl[i],sizeof(vu));
    vu P; memcpy(&P,Pl,sizeof(vu));
    w[0]=(P<<24)|(X[0]>>8);
    for(int i=1;i<8;i++) w[i]=(X[i-1]<<24)|(X[i]>>8);
    w[8]=(X[7]<<24)|vsplat(0x00800000u);
    for(int i=9;i<15;i++) w[i]=vsplat(0);
    w[15]=vsplat(264);
    #pragma GCC unroll 64
    for(int i=16;i<64;i++){
        vu s0=VROTR(w[i-15],7)^VROTR(w[i-15],18)^(w[i-15]>>3);
        vu s1=VROTR(w[i-2],17)^VROTR(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    vu a=vsplat(0x6a09e667u),b=vsplat(0xbb67ae85u),c=vsplat(0x3c6ef372u),d=vsplat(0xa54ff53au);
    vu e=vsplat(0x510e527fu),f=vsplat(0x9b05688cu),g=vsplat(0x1f83d9abu),h=vsplat(0x5be0cd19u);
    #pragma GCC unroll 64
    for(int i=0;i<64;i++){
        vu S1=VROTR(e,6)^VROTR(e,11)^VROTR(e,25), ch=(e&f)^(~e&g);
        vu t1=h+S1+ch+vsplat(SHA_K[i])+w[i];
        vu S0=VROTR(a,2)^VROTR(a,13)^VROTR(a,22), mj=(a&b)^(a&c)^(b&c);
        vu t2=S0+mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    H[0]=a+vsplat(0x6a09e667u); H[1]=b+vsplat(0xbb67ae85u); H[2]=c+vsplat(0x3c6ef372u); H[3]=d+vsplat(0xa54ff53au);
    H[4]=e+vsplat(0x510e527fu); H[5]=f+vsplat(0x9b05688cu); H[6]=g+vsplat(0x1f83d9abu); H[7]=h+vsplat(0x5be0cd19u);
    for(int i=0;i<8;i++){
        vu q=H[i];
        rw[i]=(q>>24)|((q>>8)&vsplat(0x0000FF00u))|((q<<8)&vsplat(0x00FF0000u))|(q<<24);
    }
    rw[8]=vsplat(0x80); for(int i=9;i<14;i++) rw[i]=vsplat(0); rw[14]=vsplat(256); rw[15]=vsplat(0);
    vu a1=vsplat(0x67452301u),b1=vsplat(0xEFCDAB89u),c1=vsplat(0x98BADCFEu),d1=vsplat(0x10325476u),e1=vsplat(0xC3D2E1F0u);
    vu a2=a1,b2=b1,c2=c1,d2=d1,e2=e1;
    rmd_segv<0>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<1>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<2>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<3>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    rmd_segv<4>(rw,a1,b1,c1,d1,e1,a2,b2,c2,d2,e2);
    vu o[5];
    o[0]=vsplat(0xEFCDAB89u)+c1+d2; o[1]=vsplat(0x98BADCFEu)+d1+e2; o[2]=vsplat(0x10325476u)+e1+a2;
    o[3]=vsplat(0xC3D2E1F0u)+a1+b2; o[4]=vsplat(0x67452301u)+b1+c2;
    u32 T[5][VLANES];
    for(int k=0;k<5;k++) memcpy(T[k],&o[k],sizeof(vu));
    for(int l=0;l<VLANES;l++){
        out[l].d[0]=T[0][l]|((u64)T[1][l]<<32); out[l].d[1]=T[2][l]|((u64)T[3][l]<<32);
        out[l].d[2]=T[4][l]; out[l].d[3]=0;
    }
}

// ---- Base58Check (P2PKH) et Bech32 (P2WPKH)
static const char* B58="123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
static std::string b58check_encode(unsigned char version,const unsigned char* h20){
    std::vector<unsigned char> d(25); d[0]=version; memcpy(&d[1],h20,20);
    unsigned char s1[32],s2[32]; sha256_bytes(d.data(),21,s1); sha256_bytes(s1,32,s2); memcpy(&d[21],s2,4);
    size_t zeros=0; while(zeros<d.size()&&d[zeros]==0) zeros++;
    std::vector<unsigned char> num(d.begin()+zeros,d.end()); std::string out;
    while(!num.empty()){
        std::vector<unsigned char> q; unsigned rem=0;
        for(unsigned char c:num){ unsigned cur=rem*256+c; unsigned qb=cur/58; rem=cur%58; if(!q.empty()||qb) q.push_back((unsigned char)qb); }
        out.push_back(B58[rem]); num=q;
    }
    out.append(zeros,'1'); std::reverse(out.begin(),out.end());
    return out;
}
// 0 = invalide, 1 = P2PKH valide (h20 rempli), 2 = valide mais non supporté (P2SH ...)
static int b58check_decode(const std::string& s,unsigned char* h20){
    if(s.empty()||s.size()>40) return 0;
    std::vector<unsigned char> num;                      // little-endian
    size_t zeros=0; while(zeros<s.size()&&s[zeros]=='1') zeros++;
    for(char ch:s){
        const char* p=strchr(B58,ch); if(!p||!ch) return 0;
        unsigned carry=(unsigned)(p-B58);
        for(size_t j=0;j<num.size();j++){ carry+=(unsigned)num[j]*58; num[j]=(unsigned char)carry; carry>>=8; }
        while(carry){ num.push_back((unsigned char)carry); carry>>=8; }
    }
    std::vector<unsigned char> d(zeros,0); d.insert(d.end(),num.rbegin(),num.rend());
    if(d.size()!=25) return 0;
    unsigned char s1[32],s2[32]; sha256_bytes(d.data(),21,s1); sha256_bytes(s1,32,s2);
    if(memcmp(s2,&d[21],4)!=0) return 0;
    if(d[0]!=0x00) return 2;
    memcpy(h20,&d[1],20); return 1;
}
static const char* B32="qpzry9x8gf2tvdw0s3jn54khce6mua7l";
static u32 bech32_polymod(const std::vector<unsigned char>& v){
    static const u32 G[5]={0x3b6a57b2,0x26508e6d,0x1ea119fa,0x3d4233dd,0x2a1462b3};
    u32 chk=1;
    for(unsigned char x:v){ u32 b=chk>>25; chk=((chk&0x1ffffff)<<5)^x; for(int i=0;i<5;i++) if((b>>i)&1) chk^=G[i]; }
    return chk;
}
static std::vector<unsigned char> bech32_hrp_expand(const std::string& hrp){
    std::vector<unsigned char> v; for(char c:hrp) v.push_back((unsigned char)c>>5);
    v.push_back(0); for(char c:hrp) v.push_back((unsigned char)c&31); return v;
}
static std::string bech32_encode_p2wpkh(const unsigned char* h20){
    std::vector<unsigned char> data; data.push_back(0);
    int acc=0,bits=0;
    for(int i=0;i<20;i++){ acc=(acc<<8)|h20[i]; bits+=8; while(bits>=5){ bits-=5; data.push_back((acc>>bits)&31); } }
    if(bits) data.push_back((acc<<(5-bits))&31);
    std::vector<unsigned char> v=bech32_hrp_expand("bc"); v.insert(v.end(),data.begin(),data.end());
    for(int i=0;i<6;i++) v.push_back(0);
    u32 pm=bech32_polymod(v)^1;
    for(int i=0;i<6;i++) data.push_back((pm>>(5*(5-i)))&31);
    std::string out="bc1"; for(unsigned char x:data) out.push_back(B32[x]);
    return out;
}
static int bech32_decode_addr(const std::string& s0,unsigned char* h20){
    if(s0.size()<14||s0.size()>90) return 0;
    bool lo=false,up=false; for(char c:s0){ if(c>='a'&&c<='z') lo=true; if(c>='A'&&c<='Z') up=true; }
    if(lo&&up) return 0;
    std::string s=s0; for(auto& c:s) if(c>='A'&&c<='Z') c+=32;
    size_t pos=s.rfind('1'); if(pos==std::string::npos||pos<1||s.size()-pos<7) return 0;
    std::string hrp=s.substr(0,pos); std::vector<unsigned char> data;
    for(size_t i=pos+1;i<s.size();i++){ const char* p=strchr(B32,s[i]); if(!p||!s[i]) return 0; data.push_back((unsigned char)(p-B32)); }
    std::vector<unsigned char> v=bech32_hrp_expand(hrp); v.insert(v.end(),data.begin(),data.end());
    u32 pm=bech32_polymod(v);
    if(pm!=1&&pm!=0x2bc830a3) return 0;
    if(hrp!="bc") return 0;
    data.resize(data.size()-6);
    if(data.empty()) return 0;
    if(data[0]!=0||pm!=1) return 2;                      // taproot (v1+) ... : non supporté
    std::vector<unsigned char> prog; int acc=0,bits=0;
    for(size_t i=1;i<data.size();i++){ acc=(acc<<5)|data[i]; bits+=5; if(bits>=8){ bits-=8; prog.push_back((acc>>bits)&255); } }
    if(bits>=5||((acc<<(8-bits))&255)) return 0;
    if(prog.size()!=20) return 2;                        // P2WSH (32 octets) : non supporté
    memcpy(h20,prog.data(),20); return 1;
}
// ---------------------------------------------------------------- ensemble de cibles (Bloom + liste exacte)
struct Ent { U256 x; u64 par; };                       // X de la clé publique + parité de Y (0 = préfixe 02, 1 = 03)
static_assert(sizeof(Ent)==40,"taille Ent");
static inline bool x_less(const U256& a,const U256& b){
    for(int i=3;i>=0;--i){ if(a.d[i]!=b.d[i]) return a.d[i]<b.d[i]; }
    return false;
}
static bool ent_less(const Ent& a,const Ent& b){
    if(!eq(a.x,b.x)) return x_less(a.x,b.x);
    return a.par<b.par;
}
struct TargetSet {
    std::vector<Ent> list;                              // triée, sans doublon
    std::vector<u64> bloom;                             // nblocks * 8 mots de 64 bits (blocs de 512 bits)
    u64 nblocks=0;
    u64 src_hash=0;
    bool addr=false;                                    // true : la liste contient des hash160 (adresses)
};
// Filtre de Bloom "par blocs" : 1 bloc (une ligne de cache) par test, 6 bits par clé.
// X est uniforme : on lit directement ses bits, sans fonction de hachage.
static inline bool bloom_test(const u64* bl,u64 mask,const U256& x){
    const u64* blk=bl+((x.d[0]&mask)<<3);
    u64 h=x.d[1];
    for(int j=0;j<6;j++){
        u64 p=(h>>(9*j))&511;
        if(!((blk[p>>6]>>(p&63))&1)) return false;
    }
    return true;
}
static inline void bloom_insert(u64* bl,u64 mask,const U256& x){
    u64* blk=bl+((x.d[0]&mask)<<3);
    u64 h=x.d[1];
    for(int j=0;j<6;j++){ u64 p=(h>>(9*j))&511; blk[p>>6]|=1ULL<<(p&63); }
}
static void build_bloom(TargetSet& ts){
    u64 n=ts.list.size(), need=(n+15)/16; if(need<1) need=1;
    u64 nb=1; while(nb<need) nb<<=1;
    ts.nblocks=nb; ts.bloom.assign(nb*8,0);
    for(const Ent& e:ts.list) bloom_insert(ts.bloom.data(),nb-1,e.x);
}
static void finalize_targets(TargetSet& ts){
    std::sort(ts.list.begin(),ts.list.end(),ent_less);
    ts.list.erase(std::unique(ts.list.begin(),ts.list.end(),
        [](const Ent& a,const Ent& b){ return eq(a.x,b.x)&&a.par==b.par; }),ts.list.end());
    build_bloom(ts);
}
static bool ts_has_x(const TargetSet& ts,const U256& x){
    Ent probe; probe.x=x; probe.par=0;
    auto it=std::lower_bound(ts.list.begin(),ts.list.end(),probe,ent_less);
    return it!=ts.list.end() && eq(it->x,x);
}
static bool ts_has_exact(const TargetSet& ts,const U256& x,u64 par){
    Ent probe; probe.x=x; probe.par=par;
    return std::binary_search(ts.list.begin(),ts.list.end(),probe,ent_less);
}
static std::string ent_hex(const Ent& e){ return std::string(e.par?"03":"02")+to_hex(e.x); }
static std::string ent_hash_hex(const Ent& e){
    unsigned char b[20]; unpack20(e.x,b); char s[41]; for(int i=0;i<20;i++) snprintf(s+2*i,3,"%02x",b[i]); return s;
}
static std::string ent_addr_legacy(const Ent& e){ unsigned char b[20]; unpack20(e.x,b); return b58check_encode(0x00,b); }
static std::string ent_addr_bech32(const Ent& e){ unsigned char b[20]; unpack20(e.x,b); return bech32_encode_p2wpkh(b); }
static std::string ent_text(const TargetSet& ts,const Ent& e){ return ts.addr? ent_addr_legacy(e) : ent_hex(e); }
// Adresse -> Ent (hash160 empaqueté). Retour : 0 invalide, 1 ok, 2 valide mais non supportée (P2SH, P2WSH, Taproot)
static int parse_address(const std::string& s,Ent& e){
    unsigned char h[20];
    int r=b58check_decode(s,h);
    if(r==0) r=bech32_decode_addr(s,h);
    if(r==1){ e.x=pack20(h); e.par=0; }
    return r;
}

static bool parse_pub(const std::string& s,Ent& e){
    if(s.size()!=66||s[0]!='0'||(s[1]!='2'&&s[1]!='3')) return false;
    U256 x; if(!parse_hex(s.substr(2),x)||geq(x,PRIME)) return false;
    static U256 ex; static bool init=false;
    if(!init){ ex=PRIME; add_small(ex,1); ex=shr2(ex); init=true; }
    U256 r,y,y2; fe_sqr(r,x); fe_mul(r,r,x); U256 seven={{7,0,0,0}}; fe_add(r,r,seven);
    y=fe_pow(r,ex); fe_sqr(y2,y);
    if(!eq(y2,r)) return false;                          // X n'est pas sur la courbe
    e.x=x; e.par=(s[1]=='3'); return true;
}
static void parse_targets_text(const std::string& txt,TargetSet& ts,bool addr,u64& lines,u64& rejected,u64& dups,u64& unsupported){
    lines=rejected=dups=unsupported=0; ts.addr=addr;
    size_t i=0;
    if(txt.size()>=3&&(unsigned char)txt[0]==0xEF&&(unsigned char)txt[1]==0xBB&&(unsigned char)txt[2]==0xBF) i=3;
    while(i<txt.size()){
        size_t j=txt.find('\n',i); if(j==std::string::npos) j=txt.size();
        size_t a=i,b=j;
        while(a<b&&(txt[a]==' '||txt[a]=='\t'||txt[a]=='\r')) a++;
        size_t t=a; while(t<b&&txt[t]!=' '&&txt[t]!='\t'&&txt[t]!='\r') t++;   // 1er mot de la ligne
        if(t>a){
            lines++;
            Ent e; std::string tok=txt.substr(a,t-a);
            if(addr){
                int rc=parse_address(tok,e);
                if(rc==1) ts.list.push_back(e); else if(rc==2) unsupported++; else rejected++;
            } else if(parse_pub(tok,e)) ts.list.push_back(e); else rejected++;
        }
        i=j+1;
    }
    u64 valid=ts.list.size();
    finalize_targets(ts);
    dups=valid-ts.list.size();
}
static bool read_file(const std::string& path,std::string& out){
    FILE* f=fopen(path.c_str(),"rb"); if(!f) return false;
    out.clear(); char buf[1<<16]; size_t n;
    while((n=fread(buf,1,sizeof buf,f))>0) out.append(buf,n);
    fclose(f); return true;
}
static u64 fnv64(const std::string& s){
    u64 h=1469598103934665603ULL;
    for(unsigned char c:s){ h^=c; h*=1099511628211ULL; }
    return h;
}
static u64 sum_hash(const void* p,size_t n,u64 h){
    const unsigned char* c=(const unsigned char*)p; size_t i=0;
    for(;i+8<=n;i+=8){ u64 w; memcpy(&w,c+i,8); h=(h^w)*0x9E3779B97F4A7C15ULL; h^=h>>29; }
    for(;i<n;i++){ h=(h^c[i])*0x100000001B3ULL; }
    return h;
}

// Fichier .bloom : en-tête (6 x u64) + filtre + liste exacte
static const char MAGIC[8]={'P','Z','B','L','O','O','M','2'};
static const u64 ENDIAN_TAG=0x0102030405060708ULL;
static u64 payload_sum(const TargetSet& ts){
    u64 h=0x1234567812345678ULL;
    h=sum_hash(ts.bloom.data(),ts.bloom.size()*8,h);
    h=sum_hash(ts.list.data(),ts.list.size()*sizeof(Ent),h);
    return h;
}
static bool save_bloom(const TargetSet& ts,const std::string& path){
    FILE* f=fopen(path.c_str(),"wb"); if(!f) return false;
    u64 hdr[6]={ENDIAN_TAG,(u64)ts.list.size(),ts.nblocks,ts.src_hash,payload_sum(ts),(u64)(ts.addr?1:0)};
    bool ok=fwrite(MAGIC,1,8,f)==8 && fwrite(hdr,8,6,f)==6
        && fwrite(ts.bloom.data(),8,ts.bloom.size(),f)==ts.bloom.size()
        && fwrite(ts.list.data(),sizeof(Ent),ts.list.size(),f)==ts.list.size();
    ok=(fclose(f)==0)&&ok;
    return ok;
}
static bool load_bloom(const std::string& path,TargetSet& ts,std::string& err){
    FILE* f=fopen(path.c_str(),"rb"); if(!f){ err="fichier introuvable"; return false; }
    char mg[8]; u64 hdr[6];
    if(fread(mg,1,8,f)!=8||memcmp(mg,MAGIC,8)!=0||fread(hdr,8,6,f)!=6){ fclose(f); err="format invalide"; return false; }
    if(hdr[0]!=ENDIAN_TAG){ fclose(f); err="format d'octets incompatible"; return false; }
    u64 n=hdr[1], nb=hdr[2];
    if(n==0||n>(1ULL<<32)||nb==0||(nb&(nb-1))||nb>(1ULL<<32)){ fclose(f); err="en-tête incohérent"; return false; }
    ts.nblocks=nb; ts.src_hash=hdr[3]; ts.addr=(hdr[5]==1);
    ts.bloom.assign(nb*8,0); ts.list.assign(n,Ent());
    bool ok=fread(ts.bloom.data(),8,ts.bloom.size(),f)==ts.bloom.size()
         && fread(ts.list.data(),sizeof(Ent),n,f)==n;
    char extra; if(ok && fread(&extra,1,1,f)==1) ok=false;          // fichier trop long
    fclose(f);
    if(!ok){ err="fichier tronqué ou corrompu"; return false; }
    if(payload_sum(ts)!=hdr[4]){ err="somme de contrôle invalide"; return false; }
    for(size_t i=0;i<n;i++){
        if(i&&!ent_less(ts.list[i-1],ts.list[i])){ err="liste non triée"; return false; }
        if(!bloom_test(ts.bloom.data(),nb-1,ts.list[i].x)){ err="filtre incohérent"; return false; }
    }
    return true;
}

// ================================================================ MODE VANITY : adresses P2PKH personnalisées
// Un préfixe Base58 ("1Kids") correspond à des INTERVALLES de hash160 : l'adresse est "1" + Base58(N), où N = hash160 || checksum
// (24 octets). Pour chaque longueur possible de l'adresse, les nombres N qui commencent par le préfixe forment un intervalle.
// Le GPU/CPU ne compare donc que les 64 bits de poids fort du hash160 à ces intervalles (sur-ensemble exact : aucun résultat
// perdu) ; l'hôte confirme ensuite chaque candidat par l'adresse complète (Base58Check), sensible à la casse.
#define VAN_MAXI 8
struct VanSpec {
    std::string prefix;                                  // préfixe complet, avec le '1' initial
    int nint=0; u64 lo[VAN_MAXI], hi[VAN_MAXI];          // intervalles sur les 64 bits de poids fort du hash160 (ordre des octets)
    long double prob=0;                                  // probabilité EXACTE qu'une clé donne une adresse commençant par le préfixe
    std::string out_path;                                // kangaroo_vanity.txt
    VanSpec(){ memset(lo,0,sizeof lo); memset(hi,0,sizeof hi); }
};
static inline bool van_match64(const VanSpec& v,u64 t){
    for(int i=0;i<v.nint;i++) if(t>=v.lo[i]&&t<=v.hi[i]) return true;
    return false;
}
static bool u256_mul_small(U256& a,u64 m){               // a *= m ; false en cas de dépassement de 256 bits
    u64 c=0;
    for(int i=0;i<4;i++){ u128 t=(u128)a.d[i]*m+c; a.d[i]=(u64)t; c=(u64)(t>>64); }
    return c==0;
}
static long double van_ld(const U256& a){
    long double r=0, m=1; const long double B=18446744073709551616.0L;
    for(int i=0;i<4;i++){ r+=(long double)a.d[i]*m; m*=B; }
    return r;
}
static std::string van_big(long double v){               // fmt_si jusqu'à 10^21, notation scientifique au-delà
    if(v<1e21L) return fmt_si(v);
    char b[32]; snprintf(b,sizeof b,"%.2Le",v); return b;
}
static int b58_index(char c){ const char* p=c?strchr(B58,c):nullptr; return p?(int)(p-B58):-1; }
// Construit les intervalles et la probabilité exacte. Le '1' initial est facultatif.
static bool van_build(VanSpec& v,std::string pfx,std::string& err){
    if(pfx.empty()){ err="préfixe vide"; return false; }
    for(char c:pfx) if(b58_index(c)<0){ err=std::string("caractère impossible en Base58 : '")+c+"' (0, O, I et l n'existent pas)"; return false; }
    if(pfx[0]!='1') pfx="1"+pfx;
    const std::string body=pfx.substr(1);
    if(body.empty()){ err="préfixe trop court : ajoute au moins un caractère après le 1 initial"; return false; }
    if(body.size()>33){ err="préfixe trop long : une adresse P2PKH fait au plus 34 caractères"; return false; }
    size_t k=0; while(k<body.size()&&body[k]=='1') k++;     // chaque '1' après le premier = un octet 00 en tête du hash160
    if(k>19){ err="trop de '1' consécutifs : aucun hash160 réaliste"; return false; }
    const std::string q=body.substr(k);
    const U256 Nmin=shl_u64(1,8*(23-(int)k)), Nmax=low_mask(8*(24-(int)k));
    U256 P[35]; P[0]=ONE; for(int i=1;i<35;i++){ P[i]=P[i-1]; u256_mul_small(P[i],58); }
    std::vector<std::pair<U256,U256>> iv;                    // intervalles de N (24 octets)
    if(q.empty()) iv.push_back({ZERO,Nmax});
    else{
        U256 Q=ZERO;
        for(char c:q){ u256_mul_small(Q,58); add_small(Q,(u64)b58_index(c)); }
        const int m=(int)q.size();
        for(int M=m;M<=33;M++){                              // longueur Base58 du nombre N
            U256 A=Q, B=Q; add_small(B,1);
            bool ov=false;
            for(int e=0;e<M-m;e++){ if(!u256_mul_small(A,58)||!u256_mul_small(B,58)){ ov=true; break; } }
            if(ov) break;
            U256 Hi; sub_raw(Hi,B,ONE);
            if(!geq(Nmax,A)) break;                          // les longueurs suivantes sont encore plus grandes
            if(!geq(Hi,Nmin)) continue;
            U256 L=geq(A,Nmin)?A:Nmin, H=geq(Nmax,Hi)?Hi:Nmax;
            iv.push_back({L,H});
        }
    }
    if(iv.empty()){ err="aucune adresse P2PKH ne peut commencer par ce préfixe"; return false; }
    if((int)iv.size()>VAN_MAXI){ err="préfixe trop complexe"; return false; }
    v.prefix=pfx; v.nint=0; v.prob=0;
    const long double N192=ldexpl(1.0L,192);
    for(const auto& p:iv){
        U256 w; sub_raw(w,p.second,p.first);
        v.prob+=(van_ld(w)+1.0L)/N192;
        v.lo[v.nint]=shr_n(shr_n(p.first,32),96).d[0];       // hash160 = N >> 32 ; 64 bits de poids fort = hash160 >> 96
        v.hi[v.nint]=shr_n(shr_n(p.second,32),96).d[0];
        v.nint++;
    }
    return true;
}
// Base58 d'un tableau d'octets (les octets 00 de tête donnent des '1')
static std::string b58_encode_vec(const std::vector<unsigned char>& d){
    size_t zeros=0; while(zeros<d.size()&&d[zeros]==0) zeros++;
    std::vector<unsigned char> num(d.begin()+zeros,d.end()); std::string out;
    while(!num.empty()){
        std::vector<unsigned char> q; unsigned rem=0;
        for(unsigned char c:num){ unsigned cur=rem*256+c; unsigned qb=cur/58; rem=cur%58; if(!q.empty()||qb) q.push_back((unsigned char)qb); }
        out.push_back(B58[rem]); num=q;
    }
    out.append(zeros,'1'); std::reverse(out.begin(),out.end());
    return out;
}
// WIF d'une clé privée destinée à une clé publique COMPRESSÉE (commence par K ou L)
static std::string wif_compressed(const U256& k){
    std::vector<unsigned char> d; d.push_back(0x80);
    for(int i=0;i<32;i++) d.push_back((unsigned char)(k.d[3-i/8]>>(56-8*(i%8))));
    d.push_back(0x01);
    unsigned char s1[32],s2[32]; sha256_bytes(d.data(),d.size(),s1); sha256_bytes(s1,32,s2);
    d.insert(d.end(),s2,s2+4);
    return b58_encode_vec(d);
}

// ---------------------------------------------------------------- recherche : état partagé
static const int HALF=512;                 // chaque lot teste 2*HALF clés consécutives autour d'un point central
static std::vector<U256> gx,gy;            // i*G pour i = 1..HALF
static Aff SST;                            // (2*HALF)*G : pas vers le lot suivant
static bool g_tables_built=false;

static void build_tables(){
    if(g_tables_built) return;
    gx.assign(HALF+1,ZERO); gy.assign(HALF+1,ZERO);
    Aff cur=GEN;
    for(int i=1;i<=HALF;i++){ gx[i]=cur.x; gy[i]=cur.y; if(i<HALF) cur=aff_add(cur,GEN); }
    SST=aff_add(cur,cur);                  // cur = HALF*G
    g_tables_built=true;
}

struct Found { U256 key; Ent ent; std::string pub; double t=0; bool saved=false; bool outside=false; std::string path; };
struct TState {
    std::atomic<u64> keys; std::atomic<u64> block; std::atomic<int> mode;
    SpinLock lk; bool iact; u64 iblk,ioff;                // bloc en cours et position (pour le point de reprise)
    char pad[40];
    TState():keys(0),block(0),mode(-1),iact(false),iblk(0),ioff(0){}
};
typedef std::chrono::steady_clock clk;

// Modes de parcours des blocs (un seul par lancement, tous les cœurs travaillent sur la même suite)
enum Mode { MD_SOBOL=0, MD_RANDOM=1, MD_SEQ=2, MD_MC=3, MD_PI=4, MD_E=5, MD_LN=6, MD_COUNT=7 };
static const char* MODE_NAME[MD_COUNT]={"Sobol","Aléatoire","Séquentiel","Monte Carlo","Pi","e","ln 2"};
// parties fractionnaires de pi, e et ln(2) sur 64 bits (suites de Weyl : bloc = frac(i*c) * nombre_de_blocs)
static const u64 WEYL_C[3]={0x243F6A8885A308D3ULL,0xB7E151628AED2A6BULL,0xB17217F7D1CF79ABULL};

static long double u256_to_ld(const U256& a){
    long double r=0, m=1; const long double B=18446744073709551616.0L;
    for(int i=0;i<4;i++){ r+=(long double)a.d[i]*m; m*=B; }
    return r;
}
// Ordre n de la courbe secp256k1 : toute clé privée valide est dans [1, n-1]
static const U256 CURVE_N={{0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL}};

// ---------------------------------------------------------------- masque de clé privée (-mask, -maskbin, -maskfile)
// Un masque dit quelles positions (bits) de la clé privée sont déjà connues et quelle est leur valeur ; les autres sont libres.
// L'espace de recherche se réduit à 2^(positions libres). Une liste de masques est parcourue comme un seul espace de blocs : tous
// les modes (-s, -sobol, -pi, -e, -ln, -m, aléatoire) s'appliquent à l'ensemble.
struct MaskTab { std::vector<U256> x,y; };            // table v*G pour v < 2^g : v est déposé sur les g positions libres basses du masque
struct MaskInfo {
    U256 fixmask, fixval;                             // positions connues (1) et leur valeur (0 ailleurs)
    U256 lowmask;                                     // les g positions libres les plus basses : parcourues à l'intérieur d'un bloc
    std::vector<unsigned char> lowpos, hpos;          // positions de lowmask / autres positions libres (croissantes) : l'indice de bloc les parcourt
    int nfree=0, g=0, tab=-1; bool consec=false;      // consec : positions 0..g-1 toutes libres -> bloc = 2^g clés consécutives (GPU possible)
    u64 nb=1, nb_off=0;                               // nombre de blocs du masque, indice de son premier bloc
    std::string text;
};
struct MaskSet {
    std::vector<MaskInfo> m; std::vector<MaskTab> tabs;
    long double total=0; u64 NB=0; bool all_consec=true; int nskipped=0;
};
struct RawMask { std::string text; bool bin=false; };

struct Run {
    const TargetSet* ts=nullptr;
    const VanSpec* van=nullptr; std::set<std::string> van_seen;   // mode vanity (nullptr = recherche de cibles, comportement d'origine)
    int mode=MD_RANDOM;
    std::atomic<u64> seq;                                // compteur partagé (modes séquentiel, Sobol, Pi, e, ln)
    int bits=0,nthreads=1,bsize=0,nbl=0;
    U256 start,end,blockmask;
    u64 NB=1, NBP=1;                                     // NB = nombre de blocs ; NBP = puissance de 2 >= NB (suites Sobol, Pi, ...)
    int vbits=0; U256 vhmin;                             // option -vN : 0 = blocs de 2^bsize clés ; sinon blocs alignés de 2^N clés (vhmin = start >> N)
    const MaskSet* ms=nullptr;                           // option -mask / -maskbin / -maskfile (prioritaire sur vbits)
    bool sparse=false;                                   // plus de 2^26 blocs : suite sans mémoire (pas de bitmap des blocs pris / terminés)
    bool r_fail=false;                                   // option -vN impossible (trop de blocs)
    bool is_range=false; long double total=1;            // plage personnalisée ; nombre total de clés
    std::vector<std::atomic<u64>> claim;                // blocs déjà pris par un cœur
    std::vector<std::atomic<u64>> dbits;                // blocs entièrement terminés (point de reprise)
    struct Item { u64 blk,off; };
    std::vector<Item> resume_q; SpinLock qlock;          // blocs en cours repris depuis un point de sauvegarde
    std::vector<TState> st;
    std::atomic<bool> stop; std::atomic<int> done; std::atomic<int> nfound;
    SpinLock lock; std::vector<Found> found;
    bool write_file=false, emergency=true, quiet=false;
    std::string out_main="cle_trouvee.txt", out_copy="cle_trouvee_copie.txt";   // deux copies du résultat
    std::string ckpt_path; double ckpt_every=120; u64 keys_base=0;               // point de reprise
    double time_limit=0; bool timed_out=false; double status_every=60;
    clk::time_point t0;
    Run():seq(0),stop(false),done(0),nfound(0){}
};

// décalage à gauche d'un U256 de n bits (n < 256)
static U256 shl_n(U256 a,int n){
    for(int s2=0;s2<n;s2++){ for(int k=3;k>=0;--k){ a.d[k]<<=1; if(k>0) a.d[k]|=a.d[k-1]>>63; } }
    return a;
}
static const int V_MAXBITS=62;                          // option -vN : N de 1 à 62
static u64 g_dense_max=1ULL<<26;                        // au-delà : suite de blocs « sans mémoire » (modifiable par les auto-tests)
// ---- masque : accès aux blocs
static const MaskInfo& mask_of(const Run& r,u64 blk){        // masque auquel appartient le bloc (dichotomie sur nb_off)
    const std::vector<MaskInfo>& v=r.ms->m; size_t lo=0,hi=v.size();
    while(hi-lo>1){ size_t mid=(lo+hi)/2; if(v[mid].nb_off<=blk) lo=mid; else hi=mid; }
    return v[lo];
}
static U256 mask_key(const MaskInfo& mi,u64 hidx){           // clé de base du bloc : bits connus + positions libres hautes données par l'indice
    U256 k=mi.fixval;
    for(size_t j=0;j<mi.hpos.size();j++) if((hidx>>j)&1){ int p=mi.hpos[j]; k.d[p>>6]|=1ULL<<(p&63); }
    return k;
}
static U256 mask_key_or_low(const MaskInfo& mi,const U256& k){ U256 e=k; for(int i=0;i<4;i++) e.d[i]|=mi.lowmask.d[i]; return e; }

// ---- masque : analyse d'un motif. Hexadécimal par défaut (un chiffre = 4 bits), binaire avec le préfixe 0b ou -maskbin.
// Inconnu : ? * x X .   Séparateurs ignorés : espace , : _ '   Le dernier chiffre = les bits de poids faible (positions 0..3).
static bool parse_mask(const std::string& in,bool force_bin,U256& fm,U256& fv,std::string& err){
    std::string s;
    for(char c:in){ if(c==' '||c=='\t'||c=='\r'||c=='\n'||c==','||c==':'||c=='_'||c=='\'') continue; s+=c; }
    bool bin=force_bin;
    if(s.size()>=2&&s[0]=='0'&&(s[1]=='b'||s[1]=='B')){ bin=true; s=s.substr(2); }
    else if(s.size()>=2&&s[0]=='0'&&(s[1]=='x'||s[1]=='X')){ s=s.substr(2); }
    if(s.empty()){ err="motif vide"; return false; }
    const int bpc=bin?1:4;
    if((int)s.size()*bpc>256){ err="motif trop long (256 bits au maximum)"; return false; }
    fm=ZERO; fv=ZERO;
    for(size_t i=0;i<s.size();i++){
        const char c=s[i]; const int pos=(int)(s.size()-1-i)*bpc;
        if(c=='?'||c=='*'||c=='x'||c=='X'||c=='.') continue;
        int val;
        if(bin){ if(c!='0'&&c!='1'){ err=std::string("caractère invalide '")+c+"' (binaire : 0, 1 ou ?)"; return false; } val=c-'0'; }
        else if(c>='0'&&c<='9') val=c-'0'; else if(c>='a'&&c<='f') val=c-'a'+10; else if(c>='A'&&c<='F') val=c-'A'+10;
        else { err=std::string("caractère invalide '")+c+"' (hexadécimal : 0-9, a-f ou ?)"; return false; }
        for(int b=0;b<bpc;b++){ int p=pos+b; fm.d[p>>6]|=1ULL<<(p&63); if((val>>b)&1) fv.d[p>>6]|=1ULL<<(p&63); }
    }
    return true;
}
// Table v*G (v < 2^g), la clé de v étant déposée sur les positions lowpos : parcours en code de Gray (une addition de ±2^p*G par entrée)
static void mask_table_build(MaskTab& t,const std::vector<unsigned char>& lowpos){
    const int g=(int)lowpos.size(); const u32 M=1u<<g;
    t.x.assign(M,ZERO); t.y.assign(M,ZERO);
    std::vector<Aff> Q(g),NQ(g);
    for(int j=0;j<g;j++){ Q[j]=jac_to_aff(jac_mul(shl_n(ONE,lowpos[j]),GEN)); NQ[j]=Q[j]; fe_sub(NQ[j].y,ZERO,Q[j].y); }
    Jac cur; cur.X=cur.Y=cur.Z=ZERO; cur.inf=true;
    const u32 CH=4096; std::vector<Jac> J; std::vector<u32> idx; std::vector<U256> pre(CH+1); J.reserve(CH); idx.reserve(CH);
    auto flush=[&](){
        const size_t n=J.size(); if(!n) return;
        U256 acc=ONE; pre[0]=ONE;
        for(size_t i=0;i<n;i++){ fe_mul(acc,acc,J[i].Z); pre[i+1]=acc; }
        U256 inv=fe_inv(acc);
        for(size_t i=n;i-->0;){
            U256 zi,zi2,zi3; fe_mul(zi,inv,pre[i]); fe_mul(inv,inv,J[i].Z);
            fe_sqr(zi2,zi); fe_mul(zi3,zi2,zi);
            fe_mul(t.x[idx[i]],J[i].X,zi2); fe_mul(t.y[idx[i]],J[i].Y,zi3);
        }
        J.clear(); idx.clear();
    };
    for(u32 i=1;i<M;i++){
        const int j=__builtin_ctz(i); const u32 gc=i^(i>>1); const bool set=(gc>>j)&1;
        cur=jac_madd(cur,set?Q[j]:NQ[j]); J.push_back(cur); idx.push_back(gc);
        if(J.size()==CH) flush();
    }
    flush();
}
static const int MASK_GTAB_MAX=20;                      // table v*G : 2^20 points = 64 Mo
static const size_t MASK_TAB_BUDGET=2ULL<<30;           // mémoire totale des tables
// Construit l'ensemble des blocs à partir des motifs. gover > 0 : nombre de positions libres basses parcourues dans un bloc (option -vN).
static bool mask_build(MaskSet& ms,const std::vector<RawMask>& raw,const U256& start,const U256& end,u64 gover,std::vector<std::string>& warns,std::string& err){
    ms=MaskSet();
    U256 xr; for(int k=0;k<4;k++) xr.d[k]=start.d[k]^end.d[k];
    const int t=highest_bit(xr);                          // bits au-dessus de t : identiques dans toute la plage, donc connus
    std::set<std::string> seen; std::map<std::string,int> layout; size_t budget=0; u64 NBsum=0;
    for(const RawMask& rm:raw){
        U256 fm,fv; std::string e;
        const std::string shown=rm.text.size()>70? rm.text.substr(0,67)+"..." : rm.text;
        if(!parse_mask(rm.text,rm.bin,fm,fv,e)){ warns.push_back("masque ignoré ("+shown+") : "+e); ms.nskipped++; continue; }
        bool conflict=false;
        for(int p=t+1;p<256&&!conflict;p++){
            const u64 bit=1ULL<<(p&63); const bool sv=(start.d[p>>6]&bit)!=0, kn=(fm.d[p>>6]&bit)!=0, kv=(fv.d[p>>6]&bit)!=0;
            if(kn&&kv!=sv){ conflict=true; break; }
            fm.d[p>>6]|=bit; if(sv) fv.d[p>>6]|=bit; else fv.d[p>>6]&=~bit;
        }
        if(conflict){ warns.push_back("masque ignoré ("+shown+") : incompatible avec la plage (un bit connu contredit les bits communs à toute la plage)"); ms.nskipped++; continue; }
        { std::string key((const char*)&fm,32); key.append((const char*)&fv,32);
          if(!seen.insert(key).second){ warns.push_back("masque ignoré ("+shown+") : doublon"); ms.nskipped++; continue; } }
        MaskInfo mi; mi.fixmask=fm; mi.fixval=fv; mi.lowmask=ZERO; mi.text=shown;   // U256 est un POD : lowmask doit être mis à zéro avant d'être rempli par des |=
        std::vector<unsigned char> fp;
        for(int p=0;p<256&&p<=t;p++) if(!((fm.d[p>>6]>>(p&63))&1)) fp.push_back((unsigned char)p);
        mi.nfree=(int)fp.size();
        int c=0; while(c<mi.nfree&&fp[c]==c) c++;         // suite de positions libres depuis le bit 0
        int g;
        if(gover>0) g=(int)std::min<u64>(gover,(u64)mi.nfree);
        else g=std::min(18,std::max(mi.nfree-8,std::min(mi.nfree,10)));
        bool consec=(c>=g);
        if(!consec){
            if(gover==0&&c>=10){ g=c<g?c:g; consec=true; }       // assez de positions libres basses : clés consécutives (plus rapide, GPU)
            else { const int cap=gover>0? MASK_GTAB_MAX : 12; if(g>cap) g=cap; consec=(c>=g); }
        }
        mi.g=g; mi.consec=consec;
        for(int j=0;j<mi.nfree;j++){
            const int p=fp[j];
            if(j<g){ mi.lowpos.push_back((unsigned char)p); mi.lowmask.d[p>>6]|=1ULL<<(p&63); } else mi.hpos.push_back((unsigned char)p);
        }
        const int hb=mi.nfree-g;
        if(hb>62){ warns.push_back("masque ignoré ("+shown+") : trop de positions libres ("+std::to_string(mi.nfree)+") : augmente -vN"); ms.nskipped++; continue; }
        mi.nb=1ULL<<hb;
        if(NBsum>=(1ULL<<63)||mi.nb>=(1ULL<<63)-NBsum){ err="trop de blocs au total (plus de 2^63) : augmente -vN"; return false; }
        mi.nb_off=NBsum; NBsum+=mi.nb;
        if(!mi.consec){
            const std::string lk((const char*)&mi.lowmask,32);
            auto it=layout.find(lk);
            if(it==layout.end()){
                const size_t bytes=((size_t)64)<<g;
                if(budget+bytes>MASK_TAB_BUDGET){ err="tables de points trop grandes pour la mémoire (positions libres trop dispersées) : réduis -vN"; return false; }
                budget+=bytes; MaskTab tb; mask_table_build(tb,mi.lowpos); ms.tabs.push_back(std::move(tb));
                it=layout.emplace(lk,(int)ms.tabs.size()-1).first;
            }
            mi.tab=it->second; ms.all_consec=false;
        }
        ms.total+=ldexpl(1.0L,mi.nfree);
        ms.m.push_back(std::move(mi));
    }
    ms.NB=NBsum;
    if(ms.m.empty()){ err="aucun masque utilisable"; return false; }
    return true;
}
// Comme mask_build, mais détecte automatiquement un fichier de masques écrits en binaire brut (un caractère = un bit,
// typiquement une ligne par clé, longueur quelconque, sans rapport avec un multiple de 4) sans que l'utilisateur ait à
// ajouter bin: ou 0b à chaque ligne : si l'interprétation hexadécimale par défaut ne donne AUCUN masque utilisable,
// on retente en binaire les lignes qui n'étaient pas déjà marquées bin:/0b/-maskbin, et on garde ce résultat s'il
// fonctionne. N'affecte jamais un fichier hexadécimal qui produit déjà au moins un masque utilisable (entièrement
// rétrocompatible) ; si l'essai binaire échoue aussi, l'erreur d'origine (hexadécimal) est conservée.
static bool mask_build_auto(MaskSet& ms,const std::vector<RawMask>& raw,const U256& start,const U256& end,u64 gover,
                             std::vector<std::string>& warns,std::string& err,bool& used_binary_fallback){
    used_binary_fallback=false;
    std::vector<std::string> w1; std::string e1;
    if(mask_build(ms,raw,start,end,gover,w1,e1)){ warns=w1; return true; }
    if(e1!="aucun masque utilisable"){ warns=w1; err=e1; return false; }    // autre souci (table trop grande, etc.) : ne pas le masquer
    std::vector<RawMask> raw2=raw; bool any=false;
    for(RawMask& rm:raw2) if(!rm.bin){ rm.bin=true; any=true; }             // seules les lignes non déjà forcées sont retentées en binaire
    if(!any){ warns=w1; err=e1; return false; }
    MaskSet ms2; std::vector<std::string> w2; std::string e2;
    if(!mask_build(ms2,raw2,start,end,gover,w2,e2)){ warns=w1; err=e1; return false; }  // l'essai binaire échoue aussi : garder l'erreur d'origine
    ms=std::move(ms2); warns=w2; used_binary_fallback=true;
    return true;
}
// Lecture d'un fichier de masques : un masque par ligne ; # commence un commentaire ; préfixe bin: ou 0b = binaire
static bool mask_read_file(const std::string& path,std::vector<RawMask>& out,std::string& err){
    std::string txt; if(!read_file(path,txt)){ err="impossible de lire "+path; return false; }
    size_t i=0; int n=0;
    while(i<=txt.size()){
        size_t j=txt.find('\n',i); if(j==std::string::npos) j=txt.size();
        std::string line=txt.substr(i,j-i); i=j+1;
        size_t h=line.find('#'); if(h!=std::string::npos) line=line.substr(0,h);
        size_t a=line.find_first_not_of(" \t\r"), b=line.find_last_not_of(" \t\r");
        if(a==std::string::npos) continue;
        line=line.substr(a,b-a+1);
        RawMask rm; 
        if(line.size()>4&&(line.compare(0,4,"bin:")==0||line.compare(0,4,"BIN:")==0)){ rm.bin=true; line=line.substr(4); }
        rm.text=line; out.push_back(rm); n++;
    }
    if(n==0){ err="aucun masque dans "+path; return false; }
    return true;
}

// -partmaskfile i:n : analyse de la forme "i:n" (1 <= i <= n)
static bool parse_part_spec(const char* opt,const char* v,u64& pi,u64& pn){
    std::string sv=v; size_t c=sv.find(':');
    std::string a=(c==std::string::npos)?"":sv.substr(0,c), b=(c==std::string::npos)?"":sv.substr(c+1);
    auto digits=[](const std::string& s){ return !s.empty()&&s.size()<=18&&s.find_first_not_of("0123456789")==std::string::npos; };
    if(!digits(a)||!digits(b)){ fprintf(stderr,"%s doit être de la forme i:n (ex. %s 2:70).\n",opt,opt); return false; }
    pi=strtoull(a.c_str(),nullptr,10); pn=strtoull(b.c_str(),nullptr,10);
    if(pn<1||pi<1||pi>pn){ fprintf(stderr,"%s i:n : il faut 1 <= i <= n (les parties sont numérotées de 1 à n).\n",opt); return false; }
    return true;
}
// -partmaskfile i:n : découpe la liste de masques (dans l'ordre : -mask, -maskbin, -maskfile tels que donnés) en n parties
// contiguës, sans trou ni chevauchement, de tailles égales à un masque près ; ne garde que la partie i.
// Partie i = masques d'indices [ (i-1)*M/n , i*M/n [ (division entière). Le résultat ne dépend que du contenu de la liste, de i et de n.
static bool mask_part_select(std::vector<RawMask>& masks,u64 pi,u64 pn,std::string& err){
    const u64 M=(u64)masks.size();
    if(M==0){ err="-partmaskfile nécessite au moins un masque (-mask, -maskbin ou -maskfile)."; return false; }
    const u64 lo=(u64)(((unsigned __int128)(pi-1)*M)/pn), hi=(u64)(((unsigned __int128)pi*M)/pn);
    if(hi<=lo){
        err="-partmaskfile "+std::to_string((unsigned long long)pi)+":"+std::to_string((unsigned long long)pn)+" : cette partie est vide (la liste ne contient que "
            +std::to_string((unsigned long long)M)+" masque(s) pour "+std::to_string((unsigned long long)pn)+" parties).";
        return false;
    }
    std::vector<RawMask> keep(masks.begin()+(size_t)lo,masks.begin()+(size_t)hi);
    masks.swap(keep);
    printf("Partie %llu / %llu de la liste de masques : masques %llu à %llu sur %llu (%llu masque(s) explorés ici).\n",
           (unsigned long long)pi,(unsigned long long)pn,(unsigned long long)(lo+1),(unsigned long long)hi,(unsigned long long)M,(unsigned long long)(hi-lo));
    return true;
}
// Résumé affiché au démarrage
static void mask_print_summary(const MaskSet& ms,const Run& r,bool gpu_possible){
    int fmin=1<<30,fmax=0; for(const MaskInfo& mi:ms.m){ fmin=std::min(fmin,mi.nfree); fmax=std::max(fmax,mi.nfree); }
    printf("Masque : %zu masque(s) retenu(s)%s ; positions libres : %d%s ; %s clés à tester ; %s blocs.\n",ms.m.size(),
           ms.nskipped? (" ("+std::to_string(ms.nskipped)+" ignoré(s))").c_str():"",fmin,fmax!=fmin? (" à "+std::to_string(fmax)).c_str():"",fmt_si(ms.total).c_str(),fmt_int(ms.NB).c_str());
    if(ms.m.size()<=3) for(const MaskInfo& mi:ms.m)
        printf("  %s : %d positions libres (2^%d clés), blocs de 2^%d clés, %s\n",mi.text.c_str(),mi.nfree,mi.nfree,mi.g,mi.consec?"clés consécutives (CPU/GPU)":"table de points (CPU)");
    if(!ms.all_consec) printf("Certains masques ont des positions connues parmi les bits bas : calcul sur CPU uniquement%s.\n",gpu_possible?" (les GPU ne sont pas utilisés)":"");
    if(r.sparse) printf("Plus de 2^26 blocs : suite calculée sans mémoire (-s et -sobol restent exhaustifs ; les autres modes tirent sans exclure les blocs déjà faits).\n");
}

static U256 block_start(const Run& r,u64 blk){
    if(r.ms){                                                     // masque : clé de base du bloc (bornée par le début de la plage)
        const MaskInfo& mi=mask_of(r,blk); U256 k=mask_key(mi,blk-mi.nb_off);
        return geq(k,r.start)? k : r.start;
    }
    if(r.vbits){                                                  // option -vN : blocs alignés sur 2^N (mêmes bits hauts)
        U256 t={{blk,0,0,0}}, h; add_raw(h,r.vhmin,t);
        U256 s=shl_n(h,r.vbits); if(!geq(s,r.start)) s=r.start; return s;
    }
    U256 s=r.start; add_raw(s,s,shl_u64(blk,r.bsize));
    return s;
}
static U256 block_end(const Run& r,u64 blk,const U256& s){      // fin du bloc commençant en s (bornée par la fin de la plage)
    U256 e;
    if(r.ms){ const MaskInfo& mi=mask_of(r,blk); e=mask_key_or_low(mi,mask_key(mi,blk-mi.nb_off)); }
    else if(r.vbits){ e=shl_n(shr_n(s,r.vbits),r.vbits); add_raw(e,e,low_mask(r.vbits)); }
    else add_raw(e,s,r.blockmask);
    if(!geq(r.end,e)) e=r.end;
    return e;
}
// Plage quelconque [start,end] (1 <= start <= end < n). R > 0 : option -vN (blocs alignés de 2^R clés)
static void init_run_range(Run& r,const U256& start,const U256& end,int nthreads,const TargetSet* ts,bool write_file,int mode=MD_RANDOM,bool is_range=true,u64 R=0,const MaskSet* msp=nullptr){
    r.ts=ts; r.nthreads=nthreads; r.write_file=write_file; r.mode=mode; r.seq.store(0);
    r.start=start; r.end=end; r.is_range=is_range; r.bits=highest_bit(end)+1;
    U256 sz; sub_raw(sz,end,start);                     // taille - 1
    r.total=u256_to_ld(sz)+1;
    int L=highest_bit(sz)+1; if(L<0) L=0;               // taille <= 2^L
    // Taille des blocs (indépendante du matériel : le point de reprise reste valable d'une machine à l'autre) :
    // viser ~2^16 blocs, avec des blocs de 2^10 à 2^20 clés (un lot CPU = 1024 clés) (au plus 2^26 blocs)
    int b=L-16; if(b<10) b=std::min(10,std::max(0,L-6)); if(b>20) b=20;
    if(L-b>26) b=L-26;                                  // au plus 2^26 blocs
    r.bsize=b; r.blockmask=low_mask(b);
    U256 t=sz; for(int i=0;i<b;i++){ for(int k=0;k<4;k++){ t.d[k]>>=1; if(k<3) t.d[k]|=t.d[k+1]<<63; } }
    r.NB=t.d[0]+1;                                      // = ceil(taille / 2^b)  (<= 2^26)
    r.vbits=0; r.sparse=false; r.r_fail=false; r.ms=msp;
    if(msp){                                             // option -mask / -maskbin / -maskfile (prioritaire sur -vN)
        r.NB=msp->NB; r.total=msp->total; r.sparse=(r.NB>g_dense_max);
    } else if(R>0){                                      // option -vN : blocs alignés de 2^N clés (toutes les valeurs des N bits faibles)
        if(R>(u64)V_MAXBITS) r.r_fail=true;
        else {
            const int N=(int)R; r.vhmin=shr_n(start,N);
            U256 H1; sub_raw(H1,shr_n(end,N),r.vhmin);   // nombre de blocs - 1
            if((H1.d[1]|H1.d[2]|H1.d[3])||H1.d[0]>=(1ULL<<63)) r.r_fail=true;
            else { r.vbits=N; r.NB=H1.d[0]+1; r.sparse=(r.NB>g_dense_max); }
        }
    }
    r.nbl=0; while((1ULL<<r.nbl)<r.NB) r.nbl++;
    r.NBP=1ULL<<r.nbl;
    u64 nw=r.sparse? 1 : (r.NBP+63)/64;                 // sparse : pas de bitmap (un seul mot : compteur de la suite pour le point de reprise)
    r.claim=std::vector<std::atomic<u64>>(nw);
    r.dbits=std::vector<std::atomic<u64>>(nw);
    for(u64 i=0;i<nw;i++){ r.claim[i].store(0); r.dbits[i].store(0); }
    if(!r.sparse) for(u64 bb=r.NB;bb<nw*64;bb++) r.claim[bb>>6].fetch_or(1ULL<<(bb&63));   // blocs hors plage = déjà pris
    r.resume_q.clear(); r.keys_base=0;
    r.st=std::vector<TState>(nthreads);
    r.found.clear(); r.stop=false; r.done=0; r.nfound=0;
}
// Plage d'un nombre de bits : [2^(bits-1), 2^bits - 1]
static void init_run(Run& r,int bits,int nthreads,const TargetSet* ts,bool write_file,int mode=MD_RANDOM,u64 R=0){
    U256 st=ZERO; st.d[(bits-1)>>6]|=1ULL<<((bits-1)&63);
    init_run_range(r,st,low_mask(bits),nthreads,ts,write_file,mode,false,R);
}

static inline void mark_done(Run& r,u64 blk){ if(!r.sparse) r.dbits[blk>>6].fetch_or(1ULL<<(blk&63)); }
static u64 done_blocks(const Run& r){
    if(r.sparse) return r.seq.load();                     // approximation (compteur de la suite)
    u64 nd=0; for(auto& w:r.dbits) nd+=(u64)__builtin_popcountll(w.load());
    return nd;
}

static inline bool claim_block(Run& r,u64 b){
    std::atomic<u64>& w=r.claim[b>>6]; u64 m=1ULL<<(b&63);
    if(w.load(std::memory_order_relaxed)&m) return false;
    return !(w.fetch_or(m)&m);
}
static inline u64 rev_bits(u64 v,int nb){
    u64 r=0; for(int i=0;i<nb;i++){ r=(r<<1)|(v&1); v>>=1; } return r;
}

// Générateurs d'ordre de blocs. Chaque bloc n'est donné qu'une seule fois (liste partagée "claim").
struct Gen {
    Run* r; int type; Rng rng;
    u64 ZS=1,hits=0; size_t pos=0;
    std::vector<u32> order;
    Gen(Run* rr,int t,int tid):r(rr),type(t){
        rng.seed((u64)clk::now().time_since_epoch().count()*0x9E3779B97F4A7C15ULL+(u64)tid*7919+(u64)t*104729);
        if(type==MD_MC){
            u64 NBp=r->NBP, Z=std::min<u64>(NBp,4096); ZS=NBp/Z;
            order.resize(Z); std::iota(order.begin(),order.end(),0u);
            for(size_t i=order.size();i>1;--i) std::swap(order[i-1],order[rng.next()%i]);
        }
    }
    // Balayage final : prend un bloc encore libre (garantit que toute la plage est couverte, sans bloc refait)
    bool sweep(u64& blk,u64 w0){
        u64 nw=r->claim.size();
        for(u64 j=0;j<nw;j++){
            u64 w=(w0+j)%nw; u64 cur=r->claim[w].load();
            while(~cur){
                u64 b=w*64+(u64)__builtin_ctzll(~cur);
                if(claim_block(*r,b)){ blk=b; return true; }
                cur=r->claim[w].load();
            }
        }
        return false;
    }
    // Plus de 2^26 blocs : pas de mémoire des blocs faits (la suite est calculée directement, comme -s / -sobol / Weyl / aléatoire)
    bool next_sparse(u64& blk){
        const u64 NB=r->NB; const int nbl=r->nbl;
        switch(type){
        case MD_SEQ: { u64 n=r->seq.fetch_add(1); if(n>=NB) return false; blk=n; return true; }
        case MD_SOBOL:
            for(;;){ u64 n=r->seq.fetch_add(1); if(n>=r->NBP) return false; u64 b=rev_bits(n^(n>>1),nbl); if(b<NB){ blk=b; return true; } }
        case MD_PI: case MD_E: case MD_LN: {
            const u64 C=WEYL_C[type-MD_PI];
            for(;;){ u64 i=r->seq.fetch_add(1); u64 b=nbl? ((i*C)>>(64-nbl)) : 0; if(b<NB){ blk=b; return true; } }
        }
        case MD_RANDOM:
            for(;;){ u64 b=nbl? (rng.next()>>(64-nbl)) : 0; if(b<NB){ blk=b; return true; } }
        default: break;
        }
        for(;;){                                                   // Monte Carlo : zones égales visitées dans un ordre aléatoire
            if(pos>=order.size()){ for(size_t i=order.size();i>1;--i) std::swap(order[i-1],order[rng.next()%i]); pos=0; }
            u64 b=(u64)order[pos++]*ZS+(rng.next()&(ZS-1));
            if(b<NB){ blk=b; return true; }
        }
    }
    bool next(u64& blk){
        if(r->sparse) return next_sparse(blk);
        u64 NB=r->NB;
        switch(type){
        case MD_SEQ:                                       // du début à la fin de la plage
            for(;;){
                u64 n=r->seq.fetch_add(1); if(n>=NB) return false;
                if(claim_block(*r,n)){ blk=n; return true; }
            }
        case MD_SOBOL:                                     // suite de Sobol (dimension 1, ordre de Gray) : permutation des blocs
            for(;;){
                u64 n=r->seq.fetch_add(1); if(n>=r->NBP) return false;
                u64 b=rev_bits(n^(n>>1),r->nbl);
                if(claim_block(*r,b)){ blk=b; return true; }
            }
        case MD_PI: case MD_E: case MD_LN: {                 // suite de Weyl : bloc = frac(i * c) * NB, puis balayage final
            const u64 C=WEYL_C[type-MD_PI], limit=4*r->NBP+64;
            for(;;){
                u64 i=r->seq.fetch_add(1);
                if(i>=limit) return sweep(blk,0);
                u64 b=r->nbl? ((i*C)>>(64-r->nbl)) : 0;
                if(claim_block(*r,b)){ blk=b; return true; }
            }
        }
        case MD_RANDOM:                                    // blocs tirés au hasard, sans répétition
            for(int t=0;t<16;t++){ u64 b=rng.next()&(r->NBP-1); if(claim_block(*r,b)){ blk=b; return true; } }
            return sweep(blk,rng.next()%r->claim.size());
        default: break;
        }
        for(;;){                                          // Monte Carlo : zones égales visitées dans un ordre aléatoire
            if(pos>=order.size()){
                if(pos>0&&hits==0) return false;
                for(size_t i=order.size();i>1;--i) std::swap(order[i-1],order[rng.next()%i]);
                pos=0; hits=0;
            }
            u64 base=(u64)order[pos++]*ZS;
            for(int t=0;t<8;t++){ u64 b=base+(rng.next()&(ZS-1)); if(claim_block(*r,b)){ hits++; blk=b; return true; } }
            u64 off=rng.next()&(ZS-1);
            for(u64 j=0;j<ZS;j++){ u64 b=base+((off+j)&(ZS-1)); if(claim_block(*r,b)){ hits++; blk=b; return true; } }
        }
    }
};

// ---------------------------------------------------------------- résultat trouvé
static Ent ent_of_point(bool addr,const Aff& P){
    Ent e;
    if(addr){ e.x=hash160_point_bytes(P); e.par=0; }
    else { e.x=P.x; e.par=P.y.d[0]&1; }
    return e;
}
// ---------------------------------------------------------------- enregistrement fiable des résultats
static std::string result_text(const Run& r,const Found& f){
    std::string t; char b[96];
    if(r.ts->addr){
        t+="Adresse (P2PKH)     : "+ent_addr_legacy(f.ent)+"\n";
        t+="Adresse (bech32)    : "+ent_addr_bech32(f.ent)+"\n";
        t+="Hash160             : "+ent_hash_hex(f.ent)+"\n";
    }
    t+="Clé publique        : "+f.pub+"\n";
    if(r.is_range) t+="Plage               : "+to_hex(r.start)+" -> "+to_hex(r.end)+"\n";
    else { snprintf(b,sizeof b,"Bits                : %d\n",r.bits); t+=b; }
    if(f.outside) t+="Hors plage demandée : OUI (clé valide, trouvée juste après la fin de la plage)\n";
    t+="Clé privée (hex)    : "+to_hex(f.key)+"\n";
    t+="Clé privée (décimal): "+to_dec(f.key)+"\n";
    t+="Vérification        : VALIDE\n";
    snprintf(b,sizeof b,"Temps (s)           : %.1f\n",f.t); t+=b;
    t+="----------------------------------------\n";
    return t;
}
// Ajout en fin de fichier, vidé sur le disque (flush + fsync)
static bool append_result(const std::string& path,const std::string& text){
    FILE* f=fopen(path.c_str(),"ab"); if(!f) return false;
    bool ok=fwrite(text.data(),1,text.size(),f)==text.size();
    if(fflush(f)!=0) ok=false;
#ifdef _WIN32
    if(_commit(_fileno(f))!=0) ok=false;
#else
    if(fsync(fileno(f))!=0) ok=false;
#endif
    if(fclose(f)!=0) ok=false;
    return ok;
}
// Écrit puis RELIT le fichier : le texte doit s'y trouver à la fin, sinon l'enregistrement est considéré comme raté
static bool save_checked(const std::string& path,const std::string& text){
    if(!append_result(path,text)) return false;
    std::string c; if(!read_file(path,c)) return false;
    return c.size()>=text.size() && c.compare(c.size()-text.size(),text.size(),text)==0;
}
static void record_find(Run& r,const U256& key,const Aff& P){
    r.lock.lock();
    for(const Found& f:r.found) if(eq(f.key,key)){ r.lock.unlock(); return; }
    Found f; f.key=key; f.ent=ent_of_point(r.ts->addr,P); f.pub=compress_hex(P);
    f.t=std::chrono::duration<double>(clk::now()-r.t0).count();
    f.outside=!geq(r.end,key)||!geq(key,r.start);
    f.saved=!r.write_file;
    if(r.write_file){
        std::string text=result_text(r,f), used; int ok=0;
        const std::string* names[2]={&r.out_main,&r.out_copy};
        for(int i=0;i<2;i++){
            if(names[i]->empty()) continue;
            if(save_checked(*names[i],text)){ ok++; if(!used.empty()) used+=" + "; used+=*names[i]; }
        }
        if(!ok&&r.emergency){                                  // secours : autres emplacements
            std::vector<std::string> alt;
            alt.push_back("cle_trouvee_"+std::to_string((long long)time(nullptr))+".txt");
            for(const char* ev:{"TEMP","TMP","TMPDIR","USERPROFILE","HOME"}) if(const char* d=getenv(ev)) alt.push_back(std::string(d)+"/cle_trouvee_urgence.txt");
            alt.push_back("/tmp/cle_trouvee_urgence.txt");
            for(const std::string& a2:alt) if(save_checked(a2,text)){ ok++; used=a2; break; }
        }
        f.saved=ok>0; f.path=used;
        if(!f.saved&&!r.quiet){
            fprintf(stderr,"\n\n*** ATTENTION : clé trouvée mais IMPOSSIBLE à enregistrer ! Note-la maintenant : ***\n%s\n",text.c_str());
            fflush(stderr);
        }
    }
    r.found.push_back(f);
    r.nfound.fetch_add(1);
    if(r.found.size()>=r.ts->list.size()) r.stop.store(true);
    r.lock.unlock();
}
// ---- mode vanity : confirmation exacte d'un candidat (adresse complète Base58Check, sensible à la casse) et enregistrement
static void van_verify(Run& r,const U256& k){
    static SpinLock save_mx;                                    // sérialise les écritures du fichier de résultats (portable : pas de std::mutex)
    struct SaveGuard{ SpinLock& l; explicit SaveGuard(SpinLock& x):l(x){ l.lock(); } ~SaveGuard(){ l.unlock(); } };
    const VanSpec& v=*r.van;
    if(is_zero(k)||!geq(r.end,k)||!geq(k,r.start)) return;      // hors plage (clés testées en bout de lot) : appartient à un autre bloc/une autre partie
    Aff P=aff_mul(k,GEN);
    if(P.inf) return;
    const U256 hp=hash160_point_bytes(P); unsigned char h20[20]; unpack20(hp,h20);
    const std::string addr=b58check_encode(0x00,h20);
    if(addr.compare(0,v.prefix.size(),v.prefix)!=0) return;     // faux candidat du filtre (64 bits seulement)
    const std::string hx=to_hex(k);
    r.lock.lock();
    const bool fresh=r.van_seen.insert(hx).second;
    r.lock.unlock();
    if(!fresh) return;                                           // déjà enregistrée (recouvrement de lots)
    Found f; f.key=k; f.ent=ent_of_point(true,P); f.pub=compress_hex(P);
    f.t=std::chrono::duration<double>(clk::now()-r.t0).count(); f.outside=false;
    f.saved=!r.write_file;
    if(r.write_file){
        std::string text;
        text+="Adresse (P2PKH)     : "+addr+"\n";
        text+="Clé privée (hex)    : "+hx+"\n";
        text+="WIF (compressée)    : "+wif_compressed(k)+"\n";
        text+="Clé publique        : "+f.pub+"\n";
        text+="Préfixe             : "+v.prefix+" (sensible à la casse)\n";
        text+="Plage               : "+to_hex(r.start)+" -> "+to_hex(r.end)+"\n";
        text+="Vérification        : VALIDE\n";
        text+="----------------------------------------\n";
        std::string used;
        SaveGuard save_guard(save_mx);        // une seule écriture à la fois : sinon la relecture de contrôle échoue et la clé part dans un fichier de secours
        if(save_checked(v.out_path,text)){ f.saved=true; used=v.out_path; }
        else{
            std::vector<std::string> alt;
            alt.push_back("kangaroo_vanity_"+std::to_string((long long)time(nullptr))+".txt");
            for(const char* ev:{"TMPDIR","HOME"}) if(const char* d=getenv(ev)) alt.push_back(std::string(d)+"/kangaroo_vanity_urgence.txt");
            alt.push_back("/tmp/kangaroo_vanity_urgence.txt");
            for(const std::string& a2:alt) if(save_checked(a2,text)){ f.saved=true; used=a2; break; }
            if(!f.saved&&!r.quiet){
                fprintf(stderr,"\n\n*** ATTENTION : adresse trouvée mais IMPOSSIBLE à enregistrer ! Note-la maintenant : ***\n%s\n",text.c_str());
                fflush(stderr);
            }
        }
        f.path=used;
    }
    r.lock.lock();
    r.found.push_back(f);
    r.nfound.fetch_add(1);
    r.lock.unlock();
}
// Confirmation indépendante : k*G recalculé en arithmétique affine (et hash160 par le chemin "octets"),
// puis comparaison exacte avec la liste des cibles.
static void verify_point(Run& r,const U256& k){
    if(r.van){ van_verify(r,k); return; }                       // mode vanity
    if(is_zero(k)) return;
    Aff P=aff_mul(k,GEN);
    if(P.inf) return;
    bool hit=r.ts->addr ? ts_has_exact(*r.ts,hash160_point_bytes(P),0)
                        : ts_has_exact(*r.ts,P.x,P.y.d[0]&1);
    if(hit) record_find(r,k,P);
}

struct Scratch {
    std::vector<U256> pre,cx; std::vector<char> zero; std::vector<u64> cpar; std::vector<int> coff;
    Scratch():pre(HALF+2),cx(2*HALF+2*VLANES+2),zero(HALF+2),cpar(2*HALF+2*VLANES+2),coff(2*HALF+2*VLANES+2){}
};

// Parcourt [start,end] (un bloc) par lots de 2*HALF clés consécutives.
// Un point central C sert aux clés kc-i et kc+i : une seule inversion (par lots) pour deux clés.
template<bool ADDR>
static bool search_block(Run& r,const U256& start,const U256& end,const U256& bs,Scratch& sc,TState& st){
    const TargetSet& ts=*r.ts;
    const u64* bl=ts.bloom.data(); const u64 mask=ts.nblocks-1;
    const VanSpec* vs=r.van;                                   // mode vanity (nullptr sinon)
    U256 lo=start, kc=start; add_small(kc,(u64)HALF);          // clés du lot : kc-HALF .. kc+HALF-1
    Aff C=jac_to_aff(jac_mul(kc,GEN));
    for(;;){
        if(r.stop.load(std::memory_order_relaxed)) return false;
        U256 diff; sub_raw(diff,end,lo);
        bool last=((diff.d[1]|diff.d[2]|diff.d[3])==0) && diff.d[0]<(u64)(2*HALF);
        u64 count=last? diff.d[0]+1 : (u64)(2*HALF);
        int nc=0;
        auto key_of=[&](int off){ U256 k=kc; if(off>=0) add_small(k,(u64)off); else { U256 m={{(u64)(-off),0,0,0}}; sub_raw(k,kc,m); } return k; };
        auto emit=[&](const U256& x,u64 par,int off){
            if(ADDR){ sc.cx[nc]=x; sc.cpar[nc]=par; sc.coff[nc]=off; nc++; }
            else if(bloom_test(bl,mask,x) && ts_has_x(ts,x)) verify_point(r,key_of(off));
        };
        emit(C.x,C.y.d[0]&1,0);                                   // clé centrale

        // inversion par lots de (gx[i]-C.x), i = 1..HALF, puis du pas vers le lot suivant
        U256 acc=ONE;
        for(int i=1;i<=HALF;i++){
            U256 d; fe_sub(d,gx[i],C.x);
            if(is_zero(d)){ sc.zero[i]=1; d=ONE; } else sc.zero[i]=0;
            fe_mul(acc,acc,d); sc.pre[i]=acc;
        }
        U256 dS; fe_sub(dS,SST.x,C.x);
        bool zs=is_zero(dS); if(zs) dS=ONE;
        fe_mul(acc,acc,dS);
        U256 inv=fe_inv(acc);
        Aff Cn; Cn.inf=false; Cn.x=ZERO; Cn.y=ZERO;
        {
            U256 invd; fe_mul(invd,inv,sc.pre[HALF]); fe_mul(inv,inv,dS);
            if(!zs&&!last){
                U256 t,l; fe_sub(t,SST.y,C.y); fe_mul(l,t,invd);
                fe_sqr(Cn.x,l); fe_sub(Cn.x,Cn.x,C.x); fe_sub(Cn.x,Cn.x,SST.x);
                fe_sub(t,C.x,Cn.x); fe_mul(Cn.y,l,t); fe_sub(Cn.y,Cn.y,C.y);
            }
        }
        for(int i=HALF;i>=1;--i){
            U256 invd;
            if(i>1){
                fe_mul(invd,inv,sc.pre[i-1]);
                U256 d; fe_sub(d,gx[i],C.x); if(sc.zero[i]) d=ONE;
                fe_mul(inv,inv,d);
            } else invd=inv;
            if(sc.zero[i]){                                       // cas dégénéré (C = ±i*G) : chemin lent sûr
                if(i<HALF) verify_point(r,key_of(i));
                verify_point(r,key_of(-i));
                continue;
            }
            U256 t,l,x3,y3;
            // clé kc-i : C + (-iG)
            fe_add(t,gy[i],C.y); fe_mul(l,t,invd);
            fe_sqr(x3,l); fe_sub(x3,x3,C.x); fe_sub(x3,x3,gx[i]);
            u64 par=0;
            if(ADDR){ fe_sub(t,x3,C.x); fe_mul(y3,l,t); fe_sub(y3,y3,C.y); par=y3.d[0]&1; }
            emit(x3,par,-i);
            if(i<HALF){                                           // clé kc+i : C + iG
                fe_sub(t,gy[i],C.y); fe_mul(l,t,invd);
                fe_sqr(x3,l); fe_sub(x3,x3,C.x); fe_sub(x3,x3,gx[i]);
                if(ADDR){ fe_sub(t,C.x,x3); fe_mul(y3,l,t); fe_sub(y3,y3,C.y); par=y3.d[0]&1; }
                emit(x3,par,i);
            }
        }
        if(ADDR){                                                 // hash160 par paquets de VLANES clés
            for(int c=0;c<nc;c+=VLANES){
                U256 xs[VLANES],hs[VLANES]; u64 ps[VLANES]; int m=std::min(VLANES,nc-c);
                for(int l=0;l<VLANES;l++){ int idx=c+(l<m?l:m-1); xs[l]=sc.cx[idx]; ps[l]=sc.cpar[idx]; }
                hash160_vec(xs,ps,hs);
                for(int l=0;l<m;l++){
                    if(vs){ if(van_match64(*vs,__builtin_bswap64(hs[l].d[0]))) verify_point(r,key_of(sc.coff[c+l])); }
                    else if(bloom_test(bl,mask,hs[l]) && ts_has_x(ts,hs[l])) verify_point(r,key_of(sc.coff[c+l]));
                }
            }
        }
        st.keys.fetch_add(count,std::memory_order_relaxed);
        if(last) return true;
        add_small(lo,(u64)(2*HALF)); add_small(kc,(u64)(2*HALF));
        if(zs) C=jac_to_aff(jac_mul(kc,GEN)); else C=Cn;
        {   U256 dd; sub_raw(dd,lo,bs); u64 off=(dd.d[1]|dd.d[2]|dd.d[3])? ~0ULL : dd.d[0];    // position pour le point de reprise
            st.lk.lock(); st.ioff=off; st.lk.unlock(); }
    }
}


// Bloc d'un masque « table » : clé = K | (valeur v déposée sur les g positions libres basses) ; point = K*G + T[v] (une addition affine par clé,
// une inversion par lot). Un bloc interrompu est refait en entier à la reprise.
template<bool ADDR>
static bool search_mask(Run& r,u64 blk,Scratch& sc,TState& st){
    const TargetSet& ts=*r.ts; const VanSpec* vs=r.van;
    const u64* bl=ts.bloom.data(); const u64 mask=ts.nblocks-1;
    const MaskInfo& mi=mask_of(r,blk); const MaskTab& T=r.ms->tabs[mi.tab];
    const U256 K=mask_key(mi,blk-mi.nb_off); const U256 kmax=mask_key_or_low(mi,K);
    if(!geq(kmax,r.start)||!geq(r.end,K)){ st.lk.lock(); st.ioff=0; st.lk.unlock(); return true; }   // bloc entièrement hors plage
    const bool inside=geq(K,r.start)&&geq(r.end,kmax);          // sinon : filtrage clé par clé
    const int g=mi.g; const u32 M=1u<<g;
    Aff P=jac_to_aff(jac_mul(K,GEN));                            // K*G (point à l'infini si K = 0)
    auto key_of=[&](u32 v){ U256 k=K; for(int j=0;j<g;j++) if((v>>j)&1){ int p=mi.lowpos[j]; k.d[p>>6]|=1ULL<<(p&63); } return k; };
    auto ok_range=[&](u32 v){ if(inside) return true; U256 k=key_of(v); return geq(k,r.start)&&geq(r.end,k); };
    const int BATCH=HALF; int nc=0; u64 cnt=0;
    auto flush=[&](){
        if(!ADDR){ nc=0; return; }
        for(int c=0;c<nc;c+=VLANES){
            U256 xs[VLANES],hh[VLANES]; u64 ps[VLANES]; int m=std::min(VLANES,nc-c);
            for(int l=0;l<VLANES;l++){ int idx=c+(l<m?l:m-1); xs[l]=sc.cx[idx]; ps[l]=sc.cpar[idx]; }
            hash160_vec(xs,ps,hh);
            for(int l=0;l<m;l++){
                if(vs){ if(van_match64(*vs,__builtin_bswap64(hh[l].d[0]))) verify_point(r,key_of((u32)sc.coff[c+l])); }
                else if(bloom_test(bl,mask,hh[l]) && ts_has_x(ts,hh[l])) verify_point(r,key_of((u32)sc.coff[c+l]));
            }
        }
        nc=0;
    };
    auto emit=[&](const U256& x,u64 par,u32 v){
        if(!ok_range(v)) return;
        cnt++;
        if(ADDR){ sc.cx[nc]=x; sc.cpar[nc]=par; sc.coff[nc]=(int)v; nc++; }
        else if(bloom_test(bl,mask,x) && ts_has_x(ts,x)) verify_point(r,key_of(v));
    };
    u32 v=0;
    if(!P.inf){ emit(P.x,P.y.d[0]&1,0); flush(); }               // v = 0 : la clé K elle-même
    v=1;
    if(P.inf){                                                   // K = 0 : point = T[v] directement
        for(;v<M;v++){ if(r.stop.load(std::memory_order_relaxed)) return false; emit(T.x[v],T.y[v].d[0]&1,v); if(nc>=BATCH) flush(); }
        flush();
    } else {
        for(u32 b0=1;b0<M;b0+=BATCH){
            if(r.stop.load(std::memory_order_relaxed)) return false;
            const int n=(int)std::min<u32>(BATCH,M-b0);
            U256 acc=ONE;
            for(int j=0;j<n;j++){                                // inversion par lots de (T[v].x - P.x)
                U256 d; fe_sub(d,T.x[b0+j],P.x);
                if(is_zero(d)){ sc.zero[j+1]=1; d=ONE; } else sc.zero[j+1]=0;
                fe_mul(acc,acc,d); sc.pre[j+1]=acc;
            }
            U256 inv=fe_inv(acc);
            for(int j=n-1;j>=0;--j){
                const u32 vv=b0+j; U256 invd;
                if(j>0){ fe_mul(invd,inv,sc.pre[j]); U256 d; fe_sub(d,T.x[vv],P.x); if(sc.zero[j+1]) d=ONE; fe_mul(inv,inv,d); }
                else invd=inv;
                if(sc.zero[j+1]){ if(ok_range(vv)){ cnt++; verify_point(r,key_of(vv)); } continue; }     // cas dégénéré (T[v] = ±P) : chemin lent sûr
                U256 tt,l,x3,y3; fe_sub(tt,T.y[vv],P.y); fe_mul(l,tt,invd);
                fe_sqr(x3,l); fe_sub(x3,x3,P.x); fe_sub(x3,x3,T.x[vv]);
                u64 par=0;
                if(ADDR){ fe_sub(tt,P.x,x3); fe_mul(y3,l,tt); fe_sub(y3,y3,P.y); par=y3.d[0]&1; }
                emit(x3,par,vv);
            }
            flush();
        }
    }
    st.keys.fetch_add(cnt,std::memory_order_relaxed);
    return true;
}

// ---------------------------------------------------------------- distribution des blocs et travailleurs
static bool prepare_block(const Run& r,u64 blk,u64 off,U256& s,U256& e,U256& bs){
    bs=block_start(r,blk); e=block_end(r,blk,bs); s=bs; add_small(s,off);
    return geq(e,s);
}
// Prend d'abord les blocs repris d'un point de sauvegarde, puis demande un nouveau bloc à la suite choisie
static bool next_block(Run& r,Gen& g,u64& blk,u64& off){
    r.qlock.lock();
    if(!r.resume_q.empty()){ Run::Item it=r.resume_q.back(); r.resume_q.pop_back(); r.qlock.unlock(); blk=it.blk; off=it.off; return true; }
    r.qlock.unlock();
    off=0; return g.next(blk);
}
struct WArg { Run* r; int tid; };
static void worker(WArg* a){
    Run& r=*a->r; int tid=a->tid; TState& st=r.st[tid];
    Gen g(&r,r.mode,tid);                                                   // tous les cœurs : le même mode
    Scratch sc;
    while(!r.stop.load()){
        u64 blk,off;
        if(!next_block(r,g,blk,off)) break;                                  // tous les blocs sont pris
        st.mode.store(g.type); st.block.store(blk);
        bool ok;
        if(r.ms&&!mask_of(r,blk).consec){                                    // positions libres dispersées : table de points (CPU)
            st.lk.lock(); st.iact=true; st.iblk=blk; st.ioff=0; st.lk.unlock();
            ok=r.ts->addr? search_mask<true>(r,blk,sc,st) : search_mask<false>(r,blk,sc,st);
        } else {
            U256 bs,s,e;
            if(!prepare_block(r,blk,off,s,e,bs)){ mark_done(r,blk); continue; }
            st.lk.lock(); st.iact=true; st.iblk=blk; st.ioff=off; st.lk.unlock();
            ok=r.ts->addr? search_block<true>(r,s,e,bs,sc,st) : search_block<false>(r,s,e,bs,sc,st);
        }
        if(ok){ mark_done(r,blk); st.lk.lock(); st.iact=false; st.lk.unlock(); }
    }
    st.mode.store(-1);
    r.done.fetch_add(1);
}
#ifdef _WIN32
static DWORD WINAPI thread_main(LPVOID p){ worker((WArg*)p); return 0; }
static int hw_threads(){ SYSTEM_INFO si; GetSystemInfo(&si); return (int)si.dwNumberOfProcessors; }
static void sleep_ms(int ms){ Sleep(ms); }
#else
static int hw_threads(){ int n=(int)std::thread::hardware_concurrency(); return n>0?n:1; }
static void sleep_ms(int ms){ std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
#endif

// ---------------------------------------------------------------- affichage
// ---------------------------------------------------------------- point de reprise (checkpoint)
static const char CK_MAGIC[8]={'P','Z','C','K','P','T','0','1'};
static u64 run_signature(const Run& r){
    u64 h=0xC0FFEE1234567ULL;
    h=sum_hash(r.ts->list.data(),r.ts->list.size()*sizeof(Ent),h);
    if(r.van) h=sum_hash(r.van->prefix.data(),r.van->prefix.size(),h);     // vanity : autre préfixe = autre recherche
    h=sum_hash(&r.start,sizeof(U256),h); h=sum_hash(&r.end,sizeof(U256),h);
    if(r.ms) for(const MaskInfo& mi:r.ms->m) h=sum_hash(&mi.fixmask,sizeof(U256)*2,h);   // masque : fait partie de la signature du point de reprise
    u64 ex[4]={(u64)r.vbits,(u64)r.bsize,(u64)(r.ts->addr?1:0),r.NB};
    return sum_hash(ex,sizeof ex,h);
}
static std::string ckpt_default_path(const Run& r,const std::string& dir){
    char b[64]; snprintf(b,sizeof b,"puzzle_%016llx.ckpt",(unsigned long long)run_signature(r));
    return dir+b;
}
static bool save_ckpt(Run& r,const std::string& path,u64 keys_total){
    std::vector<Run::Item> infl;
    for(int i=0;i<r.nthreads;i++){ TState& s=r.st[i]; s.lk.lock(); if(s.iact){ Run::Item it={s.iblk,s.ioff}; infl.push_back(it); } s.lk.unlock(); }
    std::vector<u64> words(r.dbits.size());                                   // lu APRÈS les blocs en cours
    for(size_t i=0;i<words.size();i++) words[i]=r.dbits[i].load();
    if(r.sparse){ const u64 sq=r.seq.load(), m=8ULL*(u64)r.nthreads; words[0]=sq>m? sq-m : 0; }   // sans mémoire : on sauve le compteur de la suite (avec une marge)
    std::vector<Run::Item> keep;
    for(const Run::Item& e:infl) if(e.blk<r.NB&&(r.sparse||!((words[e.blk>>6]>>(e.blk&63))&1))) keep.push_back(e);
    std::sort(keep.begin(),keep.end(),[](const Run::Item& a,const Run::Item& b){ return a.blk<b.blk||(a.blk==b.blk&&a.off<b.off); });
    keep.erase(std::unique(keep.begin(),keep.end(),[](const Run::Item& a,const Run::Item& b){ return a.blk==b.blk; }),keep.end());
    u64 H[16]={ENDIAN_TAG,r.start.d[0],r.start.d[1],r.start.d[2],r.start.d[3],r.end.d[0],r.end.d[1],r.end.d[2],r.end.d[3],
               (u64)r.bsize,(u64)r.vbits,r.NB,run_signature(r),(u64)words.size(),(u64)keep.size(),keys_total};
    u64 ck=sum_hash(words.data(),words.size()*8,0xABCDEF01ULL);
    ck=sum_hash(keep.data(),keep.size()*sizeof(Run::Item),ck);
    ck=sum_hash(H,sizeof H,ck);
    std::string tmp=path+".tmp";
    FILE* f=fopen(tmp.c_str(),"wb"); if(!f) return false;
    bool ok=fwrite(CK_MAGIC,1,8,f)==8&&fwrite(H,8,16,f)==16&&fwrite(&ck,8,1,f)==1
          &&fwrite(words.data(),8,words.size(),f)==words.size()
          &&(keep.empty()||fwrite(keep.data(),sizeof(Run::Item),keep.size(),f)==keep.size());
    if(fflush(f)!=0) ok=false;
#ifdef _WIN32
    if(ok&&_commit(_fileno(f))!=0) ok=false;
#else
    if(ok&&fsync(fileno(f))!=0) ok=false;
#endif
    if(fclose(f)!=0) ok=false;
    if(!ok){ remove(tmp.c_str()); return false; }
#ifdef _WIN32
    if(!MoveFileExA(tmp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)){ remove(tmp.c_str()); return false; }   // rename() n'écrase pas sous Windows
#else
    if(rename(tmp.c_str(),path.c_str())!=0){ remove(tmp.c_str()); return false; }
#endif
    return true;
}
// 1 = reprise chargée ; 0 = aucun fichier ; -1 = fichier présent mais incompatible/corrompu (ignoré)
static int load_ckpt(Run& r,const std::string& path,std::string& msg){
    FILE* f=fopen(path.c_str(),"rb"); if(!f) return 0;
    char mg[8]; u64 H[16],ck;
    if(fread(mg,1,8,f)!=8||memcmp(mg,CK_MAGIC,8)!=0||fread(H,8,16,f)!=16||fread(&ck,8,1,f)!=1){ fclose(f); msg="format invalide"; return -1; }
    const u64 nwords=H[13], ninfl=H[14];
    if(H[0]!=ENDIAN_TAG||nwords!=r.dbits.size()||ninfl>(1ULL<<32)){ fclose(f); msg="en-tête incompatible"; return -1; }
    std::vector<u64> words(nwords); std::vector<Run::Item> infl(ninfl);
    bool ok=fread(words.data(),8,nwords,f)==nwords&&(ninfl==0||fread(infl.data(),sizeof(Run::Item),ninfl,f)==ninfl);
    char extra; if(ok&&fread(&extra,1,1,f)==1) ok=false;
    fclose(f);
    if(!ok){ msg="fichier tronqué"; return -1; }
    u64 c2=sum_hash(words.data(),words.size()*8,0xABCDEF01ULL);
    c2=sum_hash(infl.data(),infl.size()*sizeof(Run::Item),c2);
    c2=sum_hash(H,sizeof H,c2);
    if(c2!=ck){ msg="somme de contrôle invalide"; return -1; }
    if(H[1]!=r.start.d[0]||H[2]!=r.start.d[1]||H[3]!=r.start.d[2]||H[4]!=r.start.d[3]||H[5]!=r.end.d[0]||H[6]!=r.end.d[1]||H[7]!=r.end.d[2]||H[8]!=r.end.d[3]
       ||H[9]!=(u64)r.bsize||H[10]!=(u64)r.vbits||H[11]!=r.NB||H[12]!=run_signature(r)){ msg="autre recherche (plage, cibles ou découpage différents)"; return -1; }
    if(r.sparse) r.seq.store(words[0]);
    else for(u64 i=0;i<nwords;i++){ r.dbits[i].store(words[i]); r.claim[i].fetch_or(words[i]); }
    r.resume_q.clear();
    for(const Run::Item& it:infl){
        if(it.blk>=r.NB||(!r.sparse&&((words[it.blk>>6]>>(it.blk&63))&1))) continue;
        U256 s,e,bs; Run::Item q=it;
        if(!prepare_block(r,it.blk,it.off,s,e,bs)) q.off=0;                    // position incohérente : on refait tout le bloc
        if(!r.sparse) r.claim[it.blk>>6].fetch_or(1ULL<<(it.blk&63));
        r.resume_q.push_back(q);
    }
    r.keys_base=H[15];
    return 1;
}

// ---------------------------------------------------------------- affichage
static bool g_ansi=false;
static std::atomic<bool> g_sigint(false);
static void on_sigint(int){ g_sigint.store(true); }
static const char* CL(const char* c){ return g_ansi?c:""; }
struct Snap {
    double el=0; u64 tested=0; double speed=0, gpu_speed=0, cpu_speed=0;
    std::vector<Found> found;
};

static void run_engine(Run& r,bool display){
    r.t0=clk::now();
    int n=r.nthreads;
    std::vector<WArg> args(n);
    for(int i=0;i<n;i++){ args[i].r=&r; args[i].tid=i; }
#ifdef _WIN32
    std::vector<HANDLE> hs(n);
    for(int i=0;i<n;i++) hs[i]=CreateThread(NULL,0,thread_main,&args[i],0,NULL);
#else
    std::vector<std::thread> ths;
    for(int i=0;i<n;i++) ths.emplace_back(worker,&args[i]);
#endif
    size_t shown=0; int lastnf=0;
    double last_status=0, last_ck=0; bool ck_warned=false;                 // le premier récapitulatif arrive après status_every secondes (défaut 60)
    auto session_keys=[&]()->u64{ u64 k=0; for(int i=0;i<n;i++) k+=r.st[i].keys.load(); return k; };
    // Affichage : AUCUN tableau, rien n'est redessiné. Chaque clé trouvée est annoncée tout de suite (et déjà enregistrée),
    // puis une seule ligne de récapitulatif est ajoutée toutes les status_every secondes (défaut 60).
    auto draw=[&](bool final_){
        const double el=std::chrono::duration<double>(clk::now()-r.t0).count();
        const double eld=el>0?el:1e-9;
        Snap s; s.el=el;
        for(int i=0;i<n;i++){
            const u64 k=r.st[i].keys.load(); s.tested+=k;
            s.cpu_speed+=k/eld;
        }
        s.speed=s.tested/eld;
        r.lock.lock(); s.found=r.found; r.lock.unlock();
        std::string out;
        for(size_t i=shown;i<s.found.size();i++){
            const Found& fd=s.found[i];
            out+=CL("\a");                                     // signal sonore à chaque nouvelle clé
            if(r.van){
                out+=std::string(CL("\x1b[1m\x1b[32m"))+"★ ADRESSE TROUVÉE  "+ent_addr_legacy(fd.ent)+CL("\x1b[0m")+"\n";
                out+="    clé privée (hex) : "+to_hex(fd.key)+"\n";
                out+="    WIF (compressée) : "+wif_compressed(fd.key)+"\n";
            } else {
                std::string pk=ent_text(*r.ts,fd.ent); if(fd.outside) pk+=" (hors plage)";
                out+=std::string(CL("\x1b[1m\x1b[32m"))+"★ CLÉ TROUVÉE  "+pk+CL("\x1b[0m")+"\n";
                out+="    clé privée (hex) : "+to_hex(fd.key)+"\n";
            }
            out+=fd.saved? std::string(CL("\x1b[32m"))+"    enregistrée dans "+fd.path+CL("\x1b[0m")+"\n"
                         : std::string(CL("\x1b[1m\x1b[31m"))+"    !!! NON ENREGISTRÉE : recopie cette clé tout de suite !!!"+CL("\x1b[0m")+"\n";
        }
        if(final_||el-last_status>=r.status_every){            // ligne de récapitulatif : éléments essentiels
            last_status=el;
            const long double all=(long double)s.tested+(long double)r.keys_base;
            long double pct=100.0L*all/r.total; if(pct>100) pct=100;
            const long double eta=(r.total-all)/(s.speed>0?s.speed:1);
            char b[256];
                snprintf(b,sizeof b,"[%6.2Lf%%] %s | %s | écoulé %s | reste ~%s | trouvées ",pct,MODE_NAME[r.mode],fmt_rate(s.speed).c_str(),fmt_time(el).c_str(),fmt_time(eta).c_str());
            out+=b;
            out+=r.van? std::to_string(s.found.size()) : std::to_string(s.found.size())+"/"+std::to_string(r.ts->list.size());
            out+="\n";
        }
        if(!out.empty()){ fputs(out.c_str(),stdout); fflush(stdout); }
        shown=s.found.size();
    };
    if(display) draw(false);
    while(r.done.load()<n){
        sleep_ms(display?100:2);
        const double el=std::chrono::duration<double>(clk::now()-r.t0).count();
        if(g_sigint.load()) r.stop.store(true);
        if(r.time_limit>0&&el>=r.time_limit&&!r.stop.load()){ r.timed_out=true; r.stop.store(true); }
        int nf=r.nfound.load();
        if(display&&(nf!=lastnf||el-last_status>=r.status_every)){ lastnf=nf; draw(false); }
        if(!r.ckpt_path.empty()&&el-last_ck>=r.ckpt_every){
            last_ck=el;
            if(!save_ckpt(r,r.ckpt_path,r.keys_base+session_keys())&&!ck_warned){ ck_warned=true; fprintf(stderr,"\nAttention : impossible d'écrire le point de reprise %s\n",r.ckpt_path.c_str()); }
        }
    }
#ifdef _WIN32
    for(int i=0;i<n;i++){ WaitForSingleObject(hs[i],INFINITE); CloseHandle(hs[i]); }
#else
    for(auto& t:ths) t.join();
#endif
    if(!r.ckpt_path.empty()) save_ckpt(r,r.ckpt_path,r.keys_base+session_keys());
    if(display){ draw(true); fflush(stdout); }
}

// ---------------------------------------------------------------- initialisation & auto-tests
static void init_curve(){
    parse_hex(GX_HEX,GEN.x); parse_hex(GY_HEX,GEN.y);
    GEN.inf=false;
}

static bool engine_test(int bits,const std::vector<u64>& keys,const std::vector<u64>& expect,int nthreads,bool addr=false,int mode=MD_RANDOM){
    TargetSet ts; ts.addr=addr;
    for(u64 k:keys){ U256 kk={{k,0,0,0}}; Aff P=aff_mul(kk,GEN); ts.list.push_back(ent_of_point(addr,P)); }
    finalize_targets(ts);
    Run r; init_run(r,bits,nthreads,&ts,false,mode);
    run_engine(r,false);
    if(r.found.size()!=expect.size()) return false;
    for(u64 k:expect){
        bool ok=false;
        for(const Found& f:r.found) if(f.key.d[0]==k&&!f.key.d[1]&&!f.key.d[2]&&!f.key.d[3]) ok=true;
        if(!ok) return false;
    }
    return true;
}


static bool range_engine_test(u64 a,u64 b,const std::vector<u64>& keys,const std::vector<u64>& expect,int nthreads,bool addr,int mode,u64 R=0,bool exhaustive=true){
    TargetSet ts; ts.addr=addr;
    for(u64 k:keys){ U256 kk={{k,0,0,0}}; ts.list.push_back(ent_of_point(addr,aff_mul(kk,GEN))); }
    finalize_targets(ts);
    Run r; U256 sa={{a,0,0,0}}, sb={{b,0,0,0}}; init_run_range(r,sa,sb,nthreads,&ts,false,mode,true,R);
    run_engine(r,false);
    if(r.found.size()!=expect.size()) return false;
    for(u64 k:expect){ bool ok=false; for(const Found& f:r.found) if(f.key.d[0]==k&&!f.key.d[1]&&!f.key.d[2]&&!f.key.d[3]) ok=true; if(!ok) return false; }
    if(R&&exhaustive){ u64 tot=0; for(auto& t:r.st) tot+=t.keys.load(); if(tot!=b-a+1) return false; }   // -vN : chaque clé de la plage comptée une fois
    return true;
}
static bool range_coverage_test(u64 a,u64 b,int mode,int ncores,u64 R=0){
    TargetSet ts; { Aff P=aff_mul(ONE,GEN); ts.list.push_back(ent_of_point(false,P)); } finalize_targets(ts);
    Run r; U256 sa={{a,0,0,0}}, sb={{b,0,0,0}}; init_run_range(r,sa,sb,ncores,&ts,false,mode,true,R);
    std::vector<Gen> gens; for(int t=0;t<ncores;t++) gens.emplace_back(&r,mode,t);
    std::vector<char> seen(r.NB,0); u64 got=0; std::vector<char> alive(ncores,1); int live=ncores;
    for(int t=0;live>0;t=(t+1)%ncores){
        if(!alive[t]) continue;
        u64 blk; if(!gens[t].next(blk)){ alive[t]=0; live--; continue; }
        if(blk>=r.NB||seen[blk]) return false;
        seen[blk]=1; got++;
    }
    // le dernier bloc doit se terminer exactement à la fin de la plage
    U256 s2=block_start(r,r.NB-1); U256 e2=block_end(r,r.NB-1,s2);
    if(r.vbits){                                           // -vN : blocs alignés sur 2^N, contigus, qui recouvrent exactement la plage
        if(!eq(block_start(r,0),r.start)||!eq(e2,r.end)) return false;
        if(r.NB<=100000){ U256 prev_e=ZERO;
            for(u64 b=0;b<r.NB;b++){ U256 s1=block_start(r,b), e1=block_end(r,b,s1);
                if(!geq(e1,s1)||!eq(shr_n(s1,r.vbits),shr_n(e1,r.vbits))) return false;
                if(b>0){ U256 nx=prev_e; add_small(nx,1); if(!eq(nx,s1)) return false; }
                prev_e=e1; } }
    }
    return got==r.NB && geq(r.end,s2);
}

// Chaque mode doit donner chaque bloc une fois et une seule (plusieurs "cœurs" simulés en alternance).
static bool coverage_test(int bits,int mode,int ncores){
    TargetSet ts; { U256 k=ONE; Aff P=aff_mul(k,GEN); ts.list.push_back(ent_of_point(false,P)); } finalize_targets(ts);
    Run r; init_run(r,bits,ncores,&ts,false,mode);
    std::vector<Gen> gens; for(int t=0;t<ncores;t++) gens.emplace_back(&r,mode,t);
    std::vector<char> seen(r.NB,0); u64 got=0; std::vector<char> alive(ncores,1); int live=ncores;
    for(int t=0;live>0;t=(t+1)%ncores){
        if(!alive[t]) continue;
        u64 b; if(!gens[t].next(b)){ alive[t]=0; live--; continue; }
        if(b>=r.NB||seen[b]) return false;
        seen[b]=1; got++;
    }
    return got==r.NB;
}

// Enregistrement : écriture + relecture, copie de secours, panne d'un fichier, panne totale (sans plantage)
static bool save_test(){
    bool ok=true;
    TargetSet ts; U256 k={{0xABCDE,0,0,0}}; ts.list.push_back(ent_of_point(false,aff_mul(k,GEN))); finalize_targets(ts);
    const char* fa="selftest_a.tmp"; const char* fb="selftest_b.tmp";
    remove(fa); remove(fb);
    {   // 1) deux fichiers valides : les deux contiennent la clé
        Run r; init_run(r,20,1,&ts,true,MD_SEQ); r.out_main=fa; r.out_copy=fb; r.emergency=false; r.quiet=true;
        run_engine(r,false);
        std::string ca,cb; read_file(fa,ca); read_file(fb,cb);
        std::string hex=to_hex(k);
        ok=ok&&r.found.size()==1&&r.found[0].saved&&ca.find(hex)!=std::string::npos&&cb.find(hex)!=std::string::npos;
    }
    remove(fa); remove(fb);
    {   // 2) fichier principal impossible : la copie sauve la clé
        Run r; init_run(r,20,1,&ts,true,MD_SEQ); r.out_main="/dossier_inexistant_xyz/a.tmp"; r.out_copy=fb; r.emergency=false; r.quiet=true;
        run_engine(r,false);
        std::string cb; read_file(fb,cb);
        ok=ok&&r.found.size()==1&&r.found[0].saved&&cb.find(to_hex(k))!=std::string::npos;
    }
    remove(fb);
    {   // 3) aucun fichier possible : signalé comme NON enregistré, sans plantage
        Run r; init_run(r,20,1,&ts,true,MD_SEQ); r.out_main="/dossier_inexistant_xyz/a.tmp"; r.out_copy="/dossier_inexistant_xyz/b.tmp"; r.emergency=false; r.quiet=true;
        run_engine(r,false);
        ok=ok&&r.found.size()==1&&!r.found[0].saved;
    }
    return ok;
}

// Reprise : un bloc terminé est ignoré, un bloc en cours reprend à sa position, le reste est cherché normalement
static bool ckpt_test(){
    const u64 A=1000,B=500000,C=900000,D=300000;
    TargetSet ts; for(u64 k:{A,B,C,D}){ U256 kk={{k,0,0,0}}; ts.list.push_back(ent_of_point(false,aff_mul(kk,GEN))); } finalize_targets(ts);
    U256 sa={{1,0,0,0}}, sb={{1ULL<<20,0,0,0}};
    const char* path="selftest.ckpt"; remove(path);
    Run r1; init_run_range(r1,sa,sb,2,&ts,false,MD_SEQ,true,0);
    const u64 ba=(A-1)>>r1.bsize, bb=(B-1)>>r1.bsize, bc=(C-1)>>r1.bsize;
    mark_done(r1,ba);
    r1.claim[ba>>6].fetch_or(1ULL<<(ba&63));
    r1.st[0].iact=true; r1.st[0].iblk=bb; r1.st[0].ioff=500;       // B est avant la position : ne doit PAS être retrouvée
    r1.st[1].iact=true; r1.st[1].iblk=bc; r1.st[1].ioff=50;        // C est après la position : doit être retrouvée
    if(!save_ckpt(r1,path,12345)) return false;
    Run r2; init_run_range(r2,sa,sb,2,&ts,false,MD_RANDOM,true,0);
    std::string msg; if(load_ckpt(r2,path,msg)!=1||r2.keys_base!=12345||r2.resume_q.size()!=2) { remove(path); return false; }
    r2.quiet=true;
    run_engine(r2,false);
    remove(path);
    bool fC=false,fD=false,other=false;
    for(const Found& f:r2.found){ if(f.key.d[0]==C) fC=true; else if(f.key.d[0]==D) fD=true; else other=true; }
    // une reprise incompatible (autre plage) doit être refusée
    Run r3; U256 sb2={{1ULL<<21,0,0,0}}; init_run_range(r3,sa,sb2,1,&ts,false,MD_SEQ,true,0);
    save_ckpt(r1,path,1); std::string m3; int rc=load_ckpt(r3,path,m3); remove(path);
    return fC&&fD&&!other&&r2.found.size()==2&&rc==-1;
}

static bool self_test(){
    bool ok=true;
    auto check=[&](bool c,const char* name){ if(!c){ fprintf(stderr,"AUTO-TEST ÉCHEC : %s\n",name); ok=false; } };
    check(on_curve(GEN),"G sur la courbe");
    U256 ex;
    Aff g2=aff_mul({{2,0,0,0}},GEN); parse_hex("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5",ex);
    check(eq(g2.x,ex),"2G");
    Aff g3=aff_mul({{3,0,0,0}},GEN); parse_hex("F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9",ex);
    check(eq(g3.x,ex),"3G");
    U256 ks[4]={{{0xDEADBEEFCAFEBABEULL,0x0123456789ABCDEFULL,0x1111222233334444ULL,0x0ABCDEF012345678ULL}},
                {{0xFFFFFFFFFFFFFFFFULL,0,0,0}},{{0,1,0,0}},{{12345,0,0xABCDULL,0}}};
    for(int i=0;i<4;i++){
        Aff a=aff_mul(ks[i],GEN), j=jac_to_aff(jac_mul(ks[i],GEN));
        check(eq(a.x,j.x)&&eq(a.y,j.y)&&on_curve(a),"affine == jacobien");
    }
    build_tables();
    for(int i=1;i<=HALF;i++){
        Aff j=jac_to_aff(jac_mul({{(u64)i,0,0,0}},GEN));
        if(!(eq(j.x,gx[i])&&eq(j.y,gy[i]))){ check(false,"table i*G"); break; }
    }
    Aff ng=jac_to_aff(jac_mul({{(u64)(2*HALF),0,0,0}},GEN));
    check(eq(ng.x,SST.x)&&eq(ng.y,SST.y),"pas 2*HALF*G");

    // arithmétique : carré dédié, inversion par chaîne, hash160 par paquets
    {
        Rng g; g.seed(777); int bad=0;
        for(int n=0;n<4000;n++){
            U256 a; for(int i=0;i<4;i++) a.d[i]=g.next();
            if(n%5==1){ a=PRIME; a.d[0]-=g.next()%3000; }
            if(n%7==2){ a=ZERO; a.d[0]=g.next()%5; }
            if(geq(a,PRIME)) sub_raw(a,a,PRIME);
            U256 s1,s2; fe_sqr(s1,a); fe_mul(s2,a,a); if(!eq(s1,s2)) bad++;
            if(n%40==0&&!is_zero(a)){
                U256 inv=fe_inv(a),pr; fe_mul(pr,inv,a); if(!eq(pr,ONE)) bad++;
                U256 e=PRIME; e.d[0]-=2; if(!eq(inv,fe_pow(a,e))) bad++;
            }
        }
        check(bad==0,"carré / inversion");
        int bh=0;
        for(int n=0;n<200;n++){
            U256 xs[VLANES],hs[VLANES]; u64 ps[VLANES];
            for(int l=0;l<VLANES;l++){ for(int i=0;i<4;i++) xs[l].d[i]=g.next(); ps[l]=g.next()&1; }
            hash160_vec(xs,ps,hs);
            for(int l=0;l<VLANES;l++) if(!eq(hs[l],hash160_fast(xs[l],ps[l]))) bh++;
        }
        check(bh==0,"hash160 par paquets");
    }
    // SHA-256 / RIPEMD-160 / adresses (vecteurs connus : clé privée 1)
    {
        unsigned char pb[33],h[20]; pub33(GEN.x,GEN.y.d[0]&1,pb);
        hash160_bytes(pb,33,h);
        U256 want; parse_hex("00000000751e76e8199196d454941c45d1b3a323f1433bd6",want);
        unsigned char wb[20]={0x75,0x1e,0x76,0xe8,0x19,0x91,0x96,0xd4,0x54,0x94,0x1c,0x45,0xd1,0xb3,0xa3,0x23,0xf1,0x43,0x3b,0xd6};
        check(memcmp(h,wb,20)==0,"hash160(G) octets");
        check(eq(hash160_fast(GEN.x,GEN.y.d[0]&1),pack20(wb)),"hash160(G) rapide");
        check(eq(hash160_fast(g2.x,g2.y.d[0]&1),hash160_point_bytes(g2)),"hash160 rapide == octets (2G)");
        check(eq(hash160_fast(g3.x,g3.y.d[0]&1),hash160_point_bytes(g3)),"hash160 rapide == octets (3G)");
        (void)want;
        Ent e; e.x=pack20(wb); e.par=0;
        check(ent_addr_legacy(e)=="1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH","encodage Base58Check");
        check(ent_addr_bech32(e)=="bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4","encodage Bech32");
        Ent d1,d2,d3; 
        check(parse_address("1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH",d1)==1&&eq(d1.x,e.x),"décodage Base58Check");
        check(parse_address("BC1QW508D6QEJXTDG4Y5R3ZARVARY0C5XW7KV8F3T4",d2)==1&&eq(d2.x,e.x),"décodage Bech32 (majuscules)");
        check(parse_address("1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMI",d3)==0,"checksum Base58 invalide");
        check(parse_address("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t5",d3)==0,"checksum Bech32 invalide");
        check(parse_address("3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy",d3)==2,"P2SH non supporté");
        check(parse_address("bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqzk5jj0",d3)==2,"Taproot non supporté");
    }
    // lecture de texte : doublons, lignes invalides, CRLF, majuscules
    {
        Aff p1=aff_mul({{5,0,0,0}},GEN), p2=aff_mul({{77,0,0,0}},GEN);
        std::string a=compress_hex(p1), b=compress_hex(p2), A=a;
        for(auto& c:A) if(c>='a'&&c<='f') c-=32;
        std::string txt=a+"\r\n\r\n"+A+"\n"+b+" # commentaire\nzzzz\n"+std::string("04")+a.substr(2)+"\n"+a.substr(0,66-1)+"\n";
        TargetSet ts; u64 l,rj,dp,un; parse_targets_text(txt,ts,false,l,rj,dp,un);
        check(ts.list.size()==2&&dp==1&&rj==3&&l==6,"lecture du fichier texte");
        // aller-retour du fichier .bloom
        ts.src_hash=12345;
        const char* tmp="selftest.tmp.bloom";
        if(save_bloom(ts,tmp)){
            TargetSet t2; std::string err;
            bool lo=load_bloom(tmp,t2,err);
            check(lo&&t2.list.size()==2&&t2.src_hash==12345&&t2.nblocks==ts.nblocks,"relecture .bloom");
            if(lo) check(ts_has_exact(t2,p1.x,p1.y.d[0]&1)&&ts_has_exact(t2,p2.x,p2.y.d[0]&1)&&!ts_has_x(t2,g2.x),"contenu .bloom");
            // corruption volontaire : doit être détectée
            FILE* f=fopen(tmp,"r+b");
            if(f){ fseek(f,-3,SEEK_END); int c=fgetc(f); fseek(f,-3,SEEK_END); fputc(c^0x55,f); fclose(f);
                   TargetSet t3; std::string e2; check(!load_bloom(tmp,t3,e2),"détection de corruption"); }
            remove(tmp);
        }
    }
    // recherches complètes sur des clés connues (cas dégénérés, multi-cibles, multi-threads)
    check(engine_test(1,{1},{1},1),"bits=1");
    check(engine_test(2,{3},{3},1),"bits=2");
    check(engine_test(3,{4,7,5},{4,7,5},2),"bits=3");
    check(engine_test(7,{100},{100},1),"bits=7");
    check(engine_test(10,{700,600,1000,5},{700,600,1000},1),"bits=10 multi-cibles (1 thread)");
    check(engine_test(10,{700,600,1000,5},{700,600,1000},4),"bits=10 multi-cibles (4 threads)");
    check(engine_test(11,{1031,2047},{1031,2047},3),"bits=11");
    check(engine_test(17,{70000,66000,131071},{70000,66000,131071},4),"bits=17");
    check(engine_test(20,{0xFFFFF,0x80000,0x9ABCD},{0xFFFFF,0x80000,0x9ABCD},4),"bits=20");
    check(engine_test(22,{3000001},{3000001},2),"bits=22");
    check(engine_test(16,{12345},{},1),"clé absente de la plage");
    check(save_test(),"enregistrement fiable des résultats");
    check(ckpt_test(),"point de reprise");
    {   // plages quelconques (taille non puissance de 2)
        bool cv=true;
        for(int m=0;m<MD_COUNT;m++) for(int nc:{1,3})
            for(auto pr:std::vector<std::pair<u64,u64>>{{1,1},{5,5},{3,70},{1000,7777},{65536,65536+100000},{12345,12345+2000000},{77,(1ULL<<25)+99}})
                cv=cv&&range_coverage_test(pr.first,pr.second,m,nc);
        check(cv,"couverture exacte de plages quelconques");
        bool ok=true;
        for(int m=0;m<MD_COUNT;m++) for(bool ad:{false,true}){
            ok=ok&&range_engine_test(1000,7777,{1000,1001,4321,7777,2},{1000,1001,4321,7777},2,ad,m);      // 2 est hors plage
            ok=ok&&range_engine_test(5,5,{5},{5},1,ad,m);
            ok=ok&&range_engine_test(100000,300000,{100000,300000,177777,50000},{100000,300000,177777},3,ad,m);
        }
        check(ok,"recherche dans des plages quelconques");
        // permutation des N derniers bits (-vN) : blocs alignés de 2^N clés, chaque clé de la plage testée une fois, tous les modes
        bool cv2=true, ok2=true;
        for(int m=0;m<MD_COUNT;m++) for(int nc:{1,3})
            for(u64 N:{1ULL,3ULL,6ULL,10ULL})
                for(auto pr:std::vector<std::pair<u64,u64>>{{1,1},{5,5},{3,70},{1000,7777},{12345,12345+50000}})
                    cv2=cv2&&range_coverage_test(pr.first,pr.second,m,nc,N);
        check(cv2,"couverture exacte avec permutation (-vN)");
        for(int m=0;m<MD_COUNT;m++) for(bool ad:{false,true}){
            ok2=ok2&&range_engine_test(1000,7777,{1000,1001,4321,7777,2},{1000,1001,4321,7777},2,ad,m,3);
            ok2=ok2&&range_engine_test(1000,7777,{1000,1001,4321,7777,2},{1000,1001,4321,7777},1,ad,m,10);
            ok2=ok2&&range_engine_test(100000,300000,{100000,300000,177777,50000},{100000,300000,177777},3,ad,m,8);
            ok2=ok2&&range_engine_test(3,5000,{3,4,5,4096,4097,5000,1},{3,4,5,4096,4097,5000},2,ad,m,12);   // 1 : sous le début de plage, donc jamais testée
        }
        check(ok2,"recherche avec permutation (-vN) : clés trouvées et plage entièrement comptée");
        {   // plus de 2^26 blocs : suite sans mémoire (seuil abaissé pour le test)
            const u64 sv=g_dense_max; g_dense_max=16; bool oks=true;
            for(int m:{(int)MD_SEQ,(int)MD_SOBOL}){                                              // exhaustifs : chaque bloc une fois, plage entièrement comptée
                for(u64 N:{3ULL,7ULL}) oks=oks&&range_coverage_test(1000,7777,m,3,N);
                for(bool ad:{false,true}){
                    oks=oks&&range_engine_test(1000,7777,{1000,1001,4321,7777,2},{1000,1001,4321,7777},2,ad,m,3);
                    oks=oks&&range_engine_test(100000,300000,{100000,300000,177777,50000},{100000,300000,177777},3,ad,m,6);
                }
            }
            for(int m:{(int)MD_RANDOM,(int)MD_PI,(int)MD_E,(int)MD_LN,(int)MD_MC}) for(bool ad:{false,true}){   // sans exclusion : on cherche jusqu'à tout trouver
                TargetSet ts; ts.addr=ad; for(u64 k:{1000ULL,4321ULL,7777ULL}){ U256 kk={{k,0,0,0}}; ts.list.push_back(ent_of_point(ad,aff_mul(kk,GEN))); }
                finalize_targets(ts); Run r; U256 sa={{1000,0,0,0}}, sb={{7777,0,0,0}}; init_run_range(r,sa,sb,2,&ts,false,m,true,3);
                r.time_limit=60; run_engine(r,false);
                oks=oks&&r.sparse&&r.found.size()==3;
            }
            g_dense_max=sv; check(oks,"plus de 2^26 blocs : suite sans mémoire (-s et -sobol exhaustifs, autres modes trouvent tout)"); }
        {   Run rj; TargetSet t1; { Aff P=aff_mul(ONE,GEN); t1.list.push_back(ent_of_point(false,P)); } finalize_targets(t1);
            U256 sa=shl_u64(1,64), sb=low_mask(66);                                              // grande plage : 2^64 .. 2^66-1
            init_run_range(rj,sa,sb,1,&t1,false,MD_SOBOL,true,10);
            bool okh=!rj.r_fail&&rj.sparse&&rj.vbits==10&&rj.NB==(3ULL<<54);
            for(int m=0;m<MD_COUNT&&okh;m++){ Run rg; init_run_range(rg,sa,sb,1,&t1,false,m,true,10); Gen g(&rg,m,0);
                for(int i=0;i<2000&&okh;i++){ u64 b; if(!g.next(b)||b>=rg.NB) okh=false; else { U256 s1=block_start(rg,b), e1=block_end(rg,b,s1); okh=okh&&geq(s1,sa)&&geq(sb,e1)&&geq(e1,s1)&&eq(shr_n(s1,10),shr_n(e1,10)); } } }
            check(okh,"grande plage avec -v10 : blocs valides pour tous les modes");
            Run rk; init_run_range(rk,sa,sb,1,&t1,false,MD_SOBOL,true,2); check(rk.r_fail,"-vN trop petit pour la plage (plus de 2^63 blocs) refusé");
            Run rm2; init_run_range(rm2,sa,sb,1,&t1,false,MD_SOBOL,true,63); check(rm2.r_fail,"-v au-delà de 62 bits refusé"); }
    }

    {   // ---------------------------------------------------------------- masque de clé privée (-mask / -maskbin / -maskfile)
        // 1) deux masques binaires qui partitionnent exactement une petite plage (positions libres contiguës depuis le bit 0)
        bool okm=true;
        {
            U256 sa={{0x1000,0,0,0}}, sb={{0x1fff,0,0,0}};
            TargetSet t1; t1.addr=false;
            U256 k1={{0x1234,0,0,0}}, k2={{0x1a9c,0,0,0}}, kdecoy={{0x5000,0,0,0}};   // 0x234 : bit 11 = 0 ; 0xa9c : bit 11 = 1 ; leurre hors plage (jamais trouvé -> pas d'arrêt anticipé)
            t1.list.push_back(ent_of_point(false,aff_mul(k1,GEN)));
            t1.list.push_back(ent_of_point(false,aff_mul(k2,GEN)));
            t1.list.push_back(ent_of_point(false,aff_mul(kdecoy,GEN)));
            finalize_targets(t1);
            std::vector<RawMask> raws={ {std::string("0")+std::string(11,'?'),true}, {std::string("1")+std::string(11,'?'),true} };
            MaskSet ms; std::vector<std::string> warns; std::string e;
            okm=okm&&mask_build(ms,raws,sa,sb,0,warns,e);
            okm=okm&&warns.empty()&&ms.m.size()==2&&ms.NB>0;
            for(int m=0;m<MD_COUNT&&okm;m++){
                Run r; init_run_range(r,sa,sb,2,&t1,false,m,true,0,&ms);
                run_engine(r,false);
                u64 tot=0; for(auto& st:r.st) tot+=st.keys.load();
                okm=okm&&r.found.size()==2&&tot==4096;                // partition exacte : 4096 clés (le leurre, hors plage, n'est jamais trouvé : pas d'arrêt anticipé)
            }
        }
        check(okm,"masque : deux masques binaires contigus couvrent exactement la plage");

        // 2) un masque aux positions libres DISPERSÉES (table de points, CPU) : recherche + comptage exact
        bool oks=true;
        {
            U256 sa=ONE, sb={{0xFFFFFF,0,0,0}};
            TargetSet t1; t1.addr=true;
            std::vector<u64> keys={0x192537ULL,0x102030ULL,0x1f2f3fULL};   // positions libres : bits 0-3, 8-11, 16-19 (fixes : 1,2,3 sur les autres nibbles)
            for(u64 k:keys){ U256 kk={{k,0,0,0}}; t1.list.push_back(ent_of_point(true,aff_mul(kk,GEN))); }
            { U256 kd={{0xAAAAAAULL,0,0,0}}; t1.list.push_back(ent_of_point(true,aff_mul(kd,GEN))); }   // leurre : nibbles fixes (1,2,3) non respectés -> jamais trouvé
            finalize_targets(t1);
            std::vector<RawMask> raws={ {"1?2?3?",false} };
            MaskSet ms; std::vector<std::string> warns; std::string e;
            oks=oks&&mask_build(ms,raws,sa,sb,0,warns,e);
            oks=oks&&warns.empty()&&ms.m.size()==1&&!ms.m[0].consec&&!ms.all_consec&&ms.NB>0;
            for(int m=0;m<MD_COUNT&&oks;m++){
                Run r; init_run_range(r,sa,sb,2,&t1,true,m,true,0,&ms);
                run_engine(r,false);
                u64 tot=0; for(auto& st:r.st) tot+=st.keys.load();
                oks=oks&&r.found.size()==3&&tot==4096;                 // 2^12 clés (3 nibbles libres) ; le leurre ne correspond jamais : pas d'arrêt anticipé
            }
        }
        check(oks,"masque : positions libres dispersées (table de points) : recherche et comptage exacts");

        // 3) masque incompatible avec la plage -> ignoré (avec avertissement), pas de plantage
        bool okc=true;
        {
            U256 sa={{0x1000,0,0,0}}, sb={{0x1fff,0,0,0}};             // bit 12 = 1 toujours sur cette plage
            std::vector<RawMask> raws={ {"f000",false}, {std::string(4,'?'),false} };  // 1er : bit12 forcé à 0 -> conflit ; 2e : valide
            MaskSet ms; std::vector<std::string> warns; std::string e;
            okc=mask_build(ms,raws,sa,sb,0,warns,e)&&ms.m.size()==1&&ms.nskipped==1&&!warns.empty();
        }
        check(okc,"masque incompatible avec la plage : ignoré proprement (pas de plantage)");

        // 4) doublon dans une liste de masques -> ignoré
        bool okd=true;
        {
            U256 sa=ONE, sb=low_mask(16);
            std::vector<RawMask> raws={ {"00??",false}, {std::string(8,'0')+std::string(8,'?'),true} };  // même masque (bits 8-15 = 0, bits 0-7 libres), deux écritures
            MaskSet ms; std::vector<std::string> warns; std::string e;
            okd=mask_build(ms,raws,sa,sb,0,warns,e)&&ms.m.size()==1&&ms.nskipped==1;
        }
        check(okd,"masque : doublon dans une liste détecté et ignoré");

        // 5) lecture d'un fichier de masques (commentaires, lignes vides, préfixe bin:)
        bool okf=true;
        {
            std::string path=std::string(getenv("TMPDIR")?getenv("TMPDIR"):"/tmp")+"/_pz_selftest_masks.txt";
            FILE* f=fopen(path.c_str(),"w");
            okf=(f!=nullptr);
            if(f){ fputs("# commentaire\n\n  d???  \nbin:1101????????????????\n",f); fclose(f); }
            std::vector<RawMask> raws; std::string e;
            okf=okf&&mask_read_file(path,raws,e)&&raws.size()==2&&!raws[0].bin&&raws[1].bin;
            remove(path.c_str());
        }
        check(okf,"masque : lecture d'un fichier (commentaires, préfixe bin:)");
    }
    for(int m=0;m<MD_COUNT;m++){                     // chaque mode : couverture exacte des blocs + recherche réelle
        bool cv=true;
        for(int b:{1,2,5,7,10,14,19,27,33,40}) for(int nc:{1,3,4}) cv=cv&&coverage_test(b,m,nc);
        check(cv,(std::string("couverture des blocs, mode ")+MODE_NAME[m]).c_str());
        check(engine_test(14,{9000,12000,16383,8192},{9000,12000,16383,8192},3,false,m),"recherche, mode clé publique");
        check(engine_test(14,{9000,12000,16383,8192},{9000,12000,16383,8192},3,true,m),"recherche, mode adresse");
    }
    {   // clés aux frontières des lots de 2*HALF clés (début, milieu, fin, lot suivant)
        std::vector<u64> ek; for(u64 o:{0,1,HALF-1,HALF,HALF+1,2*HALF-1,2*HALF,2*HALF+1,3*HALF,1000,2047}) ek.push_back(2048+o);
        check(engine_test(12,ek,ek,1),"frontières de lots (clé publique)");
        check(engine_test(12,ek,ek,3,true),"frontières de lots (adresse)");
        std::vector<u64> e2; for(u64 o:{0,255,256,511,512,100000,262143}) e2.push_back(262144+o);
        check(engine_test(19,e2,e2,2),"frontières de lots, bloc plus grand");
        check(engine_test(19,e2,e2,2,true),"frontières de lots, bloc plus grand (adresse)");
    }
    check(engine_test(10,{700,600,1000,5},{700,600,1000},1,true),"adresses bits=10 (1 thread)");
    check(engine_test(14,{9000,12000,16383,300},{9000,12000,16383},4,true),"adresses bits=14 (4 threads)");
    check(engine_test(2,{3},{3},1,true),"adresses bits=2");
    return ok;
}

// Diagnostic : si le programme s'arrête brutalement, on affiche pourquoi (et on rend le curseur).
static Run* g_run=nullptr;
static void dump_found_emergency(){
    Run* r=g_run; if(!r||r->found.empty()) return;
    fprintf(stderr,"Clés déjà trouvées (elles sont déjà enregistrées dans le fichier de résultats) :\n");
    for(const Found& f:r->found) fprintf(stderr,"  clé privée : %s   (clé publique %s)\n",to_hex(f.key).c_str(),f.pub.c_str());
    fflush(stderr);
}
static void on_terminate(){
    fputs("\x1b[?25h",stdout); fflush(stdout);
    fprintf(stderr,"\n\nERREUR : exception C++ non rattrapée, arrêt du programme.\n"); fflush(stderr);
    dump_found_emergency();
    std::_Exit(3);
}
#ifdef _WIN32
static LONG WINAPI crash_filter(EXCEPTION_POINTERS* ep){
    fputs("\x1b[?25h",stdout); fflush(stdout);
    fprintf(stderr,"\n\nERREUR FATALE : exception 0x%08lX à l'adresse %p\n"
        "Si le code est 0xC0000005 ou 0xC0000096, recompile avec :\n"
        "  g++ -O3 -march=native -mno-avx -std=c++17 -o puzzle puzzle.cpp\n",
        (unsigned long)ep->ExceptionRecord->ExceptionCode,ep->ExceptionRecord->ExceptionAddress);
    fflush(stderr);
    dump_found_emergency();
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif
static bool looks_like_pub(const std::string& s){
    return s.size()==66&&s[0]=='0'&&(s[1]=='2'||s[1]=='3');
}
static std::string lower(std::string s){ for(auto& c:s) if(c>='A'&&c<='Z') c+=32; return s; }

// Nombre de cœurs réellement utilisables (affinité + quota cgroup des conteneurs Docker/Colab/Kaggle) ; ailleurs : cœurs matériels
static int usable_cpus(){
    int n=hw_threads(); if(n<1) n=1;
#ifdef __linux__
    cpu_set_t cs; CPU_ZERO(&cs);
    if(sched_getaffinity(0,sizeof cs,&cs)==0){ int c=CPU_COUNT(&cs); if(c>0&&c<n) n=c; }
    double q=-1;
    if(FILE* f=fopen("/sys/fs/cgroup/cpu.max","r")){
        char a[64]; long long p=0; if(fscanf(f,"%63s %lld",a,&p)==2&&strcmp(a,"max")!=0&&p>0) q=atof(a)/(double)p; fclose(f);
    } else if(FILE* f1=fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us","r")){
        long long qu=-1,p=0; if(fscanf(f1,"%lld",&qu)==1&&qu>0){ if(FILE* f2=fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us","r")){ if(fscanf(f2,"%lld",&p)==1&&p>0) q=(double)qu/(double)p; fclose(f2); } } fclose(f1);
    }
    if(q>0){ int c=(int)std::ceil(q-1e-9); if(c<1) c=1; if(c<n) n=c; }
#endif
    return n<1?1:n;
}
// Durée : "3600", "90m", "11h30m", "45s"
static bool parse_duration(const std::string& s,double& out){
    if(s.empty()) return false;
    double tot=0; size_t i=0;
    while(i<s.size()){
        size_t j=i; while(j<s.size()&&(isdigit((unsigned char)s[j])||s[j]=='.')) j++;
        if(j==i) return false;
        double v=atof(s.substr(i,j-i).c_str()); double m=1;
        if(j<s.size()){ char u=s[j]; if(u=='h'||u=='H') m=3600; else if(u=='m'||u=='M') m=60; else if(u=='s'||u=='S') m=1; else return false; j++; }
        tot+=v*m; i=j;
    }
    out=tot; return tot>0;
}
// ---------------------------------------------------------------- puzzles Bitcoin 1 à 80
// Puzzle n°N : la clé privée a exactement N bits, soit dans [2^(N-1), 2^N - 1].
static const int PUZZLE_MAX=80;
static const char* const PUZZLE_ADDR[PUZZLE_MAX+1]={
    nullptr,
    "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH",   // 1
    "1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb",   // 2
    "19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA",   // 3
    "1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e",   // 4
    "1E6NuFjCi27W5zoXg8TRdcSRq84zJeBW3k",   // 5
    "1PitScNLyp2HCygzadCh7FveTnfmpPbfp8",   // 6
    "1McVt1vMtCC7yn5b9wgX1833yCcLXzueeC",   // 7
    "1M92tSqNmQLYw33fuBvjmeadirh1ysMBxK",   // 8
    "1CQFwcjw1dwhtkVWBttNLDtqL7ivBonGPV",   // 9
    "1LeBZP5QCwwgXRtmVUvTVrraqPUokyLHqe",   // 10
    "1PgQVLmst3Z314JrQn5TNiys8Hc38TcXJu",   // 11
    "1DBaumZxUkM4qMQRt2LVWyFJq5kDtSZQot",   // 12
    "1Pie8JkxBT6MGPz9Nvi3fsPkr2D8q3GBc1",   // 13
    "1ErZWg5cFCe4Vw5BzgfzB74VNLaXEiEkhk",   // 14
    "1QCbW9HWnwQWiQqVo5exhAnmfqKRrCRsvW",   // 15
    "1BDyrQ6WoF8VN3g9SAS1iKZcPzFfnDVieY",   // 16
    "1HduPEXZRdG26SUT5Yk83mLkPyjnZuJ7Bm",   // 17
    "1GnNTmTVLZiqQfLbAdp9DVdicEnB5GoERE",   // 18
    "1NWmZRpHH4XSPwsW6dsS3nrNWfL1yrJj4w",   // 19
    "1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum",   // 20
    "14oFNXucftsHiUMY8uctg6N487riuyXs4h",   // 21
    "1CfZWK1QTQE3eS9qn61dQjV89KDjZzfNcv",   // 22
    "1L2GM8eE7mJWLdo3HZS6su1832NX2txaac",   // 23
    "1rSnXMr63jdCuegJFuidJqWxUPV7AtUf7",   // 24
    "15JhYXn6Mx3oF4Y7PcTAv2wVVAuCFFQNiP",   // 25
    "1JVnST957hGztonaWK6FougdtjxzHzRMMg",   // 26
    "128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k",   // 27
    "12jbtzBb54r97TCwW3G1gCFoumpckRAPdY",   // 28
    "19EEC52krRUK1RkUAEZmQdjTyHT7Gp1TYT",   // 29
    "1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps",   // 30
    "1LhE6sCTuGae42Axu1L1ZB7L96yi9irEBE",   // 31
    "1FRoHA9xewq7DjrZ1psWJVeTer8gHRqEvR",   // 32
    "187swFMjz1G54ycVU56B7jZFHFTNVQFDiu",   // 33
    "1PWABE7oUahG2AFFQhhvViQovnCr4rEv7Q",   // 34
    "1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb",   // 35
    "1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1",   // 36
    "14iXhn8bGajVWegZHJ18vJLHhntcpL4dex",   // 37
    "1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2",   // 38
    "122AJhKLEfkFBaGAd84pLp1kfE7xK3GdT8",   // 39
    "1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv",   // 40
    "1L5sU9qvJeuwQUdt4y1eiLmquFxKjtHr3E",   // 41
    "1E32GPWgDyeyQac4aJxm9HVoLrrEYPnM4N",   // 42
    "1PiFuqGpG8yGM5v6rNHWS3TjsG6awgEGA1",   // 43
    "1CkR2uS7LmFwc3T2jV8C1BhWb5mQaoxedF",   // 44
    "1NtiLNGegHWE3Mp9g2JPkgx6wUg4TW7bbk",   // 45
    "1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP",   // 46
    "1Pd8VvT49sHKsmqrQiP61RsVwmXCZ6ay7Z",   // 47
    "1DFYhaB2J9q1LLZJWKTnscPWos9VBqDHzv",   // 48
    "12CiUhYVTTH33w3SPUBqcpMoqnApAV4WCF",   // 49
    "1MEzite4ReNuWaL5Ds17ePKt2dCxWEofwk",   // 50
    "1NpnQyZ7x24ud82b7WiRNvPm6N8bqGQnaS",   // 51
    "15z9c9sVpu6fwNiK7dMAFgMYSK4GqsGZim",   // 52
    "15K1YKJMiJ4fpesTVUcByoz334rHmknxmT",   // 53
    "1KYUv7nSvXx4642TKeuC2SNdTk326uUpFy",   // 54
    "1LzhS3k3e9Ub8i2W1V8xQFdB8n2MYCHPCa",   // 55
    "17aPYR1m6pVAacXg1PTDDU7XafvK1dxvhi",   // 56
    "15c9mPGLku1HuW9LRtBf4jcHVpBUt8txKz",   // 57
    "1Dn8NF8qDyyfHMktmuoQLGyjWmZXgvosXf",   // 58
    "1HAX2n9Uruu9YDt4cqRgYcvtGvZj1rbUyt",   // 59
    "1Kn5h2qpgw9mWE5jKpk8PP4qvvJ1QVy8su",   // 60
    "1AVJKwzs9AskraJLGHAZPiaZcrpDr1U6AB",   // 61
    "1Me6EfpwZK5kQziBwBfvLiHjaPGxCKLoJi",   // 62
    "1NpYjtLira16LfGbGwZJ5JbDPh3ai9bjf4",   // 63
    "16jY7qLJnxb7CHZyqBP8qca9d51gAjyXQN",   // 64
    "18ZMbwUFLMHoZBbfpCjUJQTCMCbktshgpe",   // 65
    "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so",   // 66
    "1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9",   // 67
    "1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ",   // 68
    "19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG",   // 69
    "19YZECXj3SxEZMoUeJ1yiPsw8xANe7M7QR",   // 70
    "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU",   // 71
    "1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR",   // 72
    "12VVRNPi4SJqUTsp6FmqDqY5sGosDtysn4",   // 73
    "1FWGcVDK3JGzCC3WtkYetULPszMaK2Jksv",   // 74
    "1J36UjUByGroXcCvmj13U6uwaVv9caEeAt",   // 75
    "1DJh2eHFYQfACPmrvpyWc8MSTYKh7w9eRF",   // 76
    "1Bxk4CQdqL9p22JEtDfdXMsng1XacifUtE",   // 77
    "15qF6X51huDjqTmF9BJgxXdt1xcj46Jmhb",   // 78
    "1ARk8HWJMn8js8tQmGUJeQHjSE7KRkn2t8",   // 79
    "1BCf6rHUW6m3iH2ptsvnjgLruAiPQQepLe",   // 80
};

// -pN (ou -p N) est remplacé par  <adresse du puzzle N> <N> address  : cible et plage s'ajustent seules.
// Retour : 0 = pas de -p ; N = puzzle choisi (argc/argv réécrits) ; -1 = erreur (message déjà affiché).
static int expand_puzzle_arg(int& argc,char**& argv,std::vector<std::string>& store,std::vector<char*>& ptrs){
    int num=0, pos=-1, span=1;
    for(int i=1;i<argc;i++){
        std::string o=lower(argv[i]); int n=0, sp=1;
        if(o.size()>2&&o.compare(0,2,"-p")==0&&o.find_first_not_of("0123456789",2)==std::string::npos){
            n=(o.size()-2<=4)?atoi(o.c_str()+2):0;
        } else if(o=="-p"){
            std::string nx=(i+1<argc)?argv[i+1]:"";
            if(nx.empty()||nx.size()>4||nx.find_first_not_of("0123456789")!=std::string::npos){
                fprintf(stderr,"-p doit être suivi du numéro du puzzle (ex. -p66 ou -p 66).\n"); return -1; }
            n=atoi(nx.c_str()); sp=2;
        } else continue;                                           // (-pi, -pN seulement ; les autres options passent)
        if(pos>=0){ fprintf(stderr,"Un seul puzzle à la fois (-p indiqué plusieurs fois).\n"); return -1; }
        if(n<1||n>PUZZLE_MAX){ fprintf(stderr,"Puzzle inconnu : le numéro doit être compris entre 1 et %d.\n",PUZZLE_MAX); return -1; }
        num=n; pos=i; span=sp;
    }
    if(!num) return 0;
    store.clear(); store.push_back(argv[0]);
    store.push_back(PUZZLE_ADDR[num]); store.push_back(std::to_string(num)); store.push_back("address");
    for(int i=1;i<argc;i++) if(i<pos||i>=pos+span) store.push_back(argv[i]);
    ptrs.clear(); for(auto& s:store) ptrs.push_back(const_cast<char*>(s.c_str())); ptrs.push_back(nullptr);
    argc=(int)store.size(); argv=ptrs.data();
    printf("Puzzle n°%d : %s  (clé de %d bits : 2^%d … 2^%d - 1)\n",num,PUZZLE_ADDR[num],num,num-1,num);
    return num;
}

static void usage(const char* prog);
// ================================================================ MODE VANITY : lancement
static std::string van_clean_hex(std::string h){ if(h.size()>=2&&h[0]=='0'&&(h[1]=='x'||h[1]=='X')) h=h.substr(2); return h; }

// Auto-tests du mode vanity : intervalles, probabilité exacte, adresse/WIF connus, moteur CPU (plage, recouvrements)
static bool van_selftest(){
    bool ok=true;
    auto check=[&](bool c,const char* name){ if(!c){ fprintf(stderr,"AUTO-TEST ÉCHEC (vanity) : %s\n",name); ok=false; } };
    std::string e;
    {   // vecteurs connus : clé privée 1
        U256 one=ONE; Aff P=aff_mul(one,GEN); U256 hp=hash160_point_bytes(P); unsigned char h[20]; unpack20(hp,h);
        check(b58check_encode(0x00,h)=="1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH","adresse de la clé 1");
        check(wif_compressed(one)=="KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgd9M7rFU73sVHnoWn","WIF de la clé 1");
        VanSpec v1,v2,v3;
        check(van_build(v1,"BgGZ9t",e)&&van_match64(v1,__builtin_bswap64(hp.d[0])),"intervalle : préfixe sans '1' initial");
        check(van_build(v2,"1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH",e)&&van_match64(v2,__builtin_bswap64(hp.d[0])),"intervalle : adresse complète");
        check(!van_build(v3,"1Kid0",e),"préfixe invalide (0) refusé");
        check(!van_build(v3,"1",e),"préfixe '1' seul refusé");
    }
    Rng rng; rng.seed(0x5EED5EEDULL);
    auto rand_h20=[&](unsigned char* h,int zeros){ for(int i=0;i<20;i++) h[i]=(unsigned char)(rng.next()>>24); if(zeros<0) return;          // zeros<0 : hash160 totalement aléatoire
        for(int i=0;i<zeros;i++) h[i]=0;                                       // sinon : exactement `zeros` octets 00 en tête
        if(h[zeros]==0) h[zeros]=1; };
    {   // probabilité exacte + intervalles = sur-ensemble du préfixe (hash160 aléatoires, longueurs d'adresse variables)
        const char* pf[]={"1A","1Ab","1Z","11","1zz","1KiDs"}; const int S=60000;
        for(const char* p:pf){
            VanSpec v; if(!van_build(v,p,e)){ check(false,"construction du préfixe"); continue; }
            int cp=0, ci=0; bool impl=true;
            for(int s=0;s<S;s++){
                unsigned char h[20]; rand_h20(h,-1);
                std::string a=b58check_encode(0x00,h); bool pm=a.compare(0,v.prefix.size(),v.prefix)==0;
                U256 hp=pack20(h); bool im=van_match64(v,__builtin_bswap64(hp.d[0]));
                if(pm){ cp++; if(!im) impl=false; }
                if(im) ci++;
            }
            const double ex=(double)S*(double)v.prob, tol=5*std::sqrt(ex)+3;
            if(!impl||std::fabs(cp-ex)>tol||ci<cp||ci>cp+3){
                fprintf(stderr,"  préfixe %s : attendu %.1f, observé %d, candidats %d, implication %s\n",p,ex,cp,ci,impl?"ok":"VIOLÉE"); ok=false; }
        }
    }
    {   // le préfixe d'une adresse réelle (toutes longueurs, avec octets 00 en tête) est toujours reconnu
        bool all=true;
        for(int z=0;z<4;z++) for(int s=0;s<600;s++){
            unsigned char h[20]; rand_h20(h,z); std::string a=b58check_encode(0x00,h); U256 hp=pack20(h);
            size_t L=2+(size_t)(rng.next()%11); if(L>a.size()) L=a.size();
            VanSpec v; if(!van_build(v,a.substr(0,L),e)||!van_match64(v,__builtin_bswap64(hp.d[0]))) all=false;
        }
        check(all,"préfixes d'adresses réelles reconnus");
    }
    {   // moteur : plage quelconque, recouvrements de lots, déduplication, CPU
        const u64 A=1000,B=4999,T=3333;
        U256 kt={{T,0,0,0}}; Aff Pt=aff_mul(kt,GEN); U256 ht=hash160_point_bytes(Pt); unsigned char hb[20]; unpack20(ht,hb);
        const std::string at=b58check_encode(0x00,hb);
        VanSpec v; check(van_build(v,at.substr(0,3),e),"préfixe du moteur");
        std::set<u64> expA,expB;
        for(u64 k=A;k<=B;k++){
            U256 kk={{k,0,0,0}}; Aff P=aff_mul(kk,GEN); U256 hp=hash160_point_bytes(P); unsigned char h[20]; unpack20(hp,h);
            if(b58check_encode(0x00,h).compare(0,v.prefix.size(),v.prefix)==0){ expA.insert(k); if(k<=T-1) expB.insert(k); }
        }
        check(expA.count(T)==1,"clé de test dans la plage");
        auto run_van=[&](u64 a,u64 b,int mode,int ncpu,std::set<u64>& out,size_t& nf)->bool{
            TargetSet ts; ts.addr=true; ts.nblocks=1; ts.bloom.assign(8,0);
            Run r; U256 sa={{a,0,0,0}}, sb={{b,0,0,0}}; init_run_range(r,sa,sb,ncpu,&ts,false,mode,true,0);
            r.van=&v; r.quiet=true;
            run_engine(r,false);
            nf=r.found.size(); for(const Found& f:r.found) out.insert(f.key.d[0]);
            return true;
        };
        const int modes[3]={MD_SEQ,MD_RANDOM,MD_SOBOL};
        for(int mi=0;mi<3;mi++){
            std::set<u64> got; size_t nf=0;
            bool r1=run_van(A,B,modes[mi],2,got,nf);
            check(r1&&got==expA&&nf==expA.size(),"vanity CPU : plage complète, sans doublon");
            std::set<u64> got2; size_t nf2=0;
            bool r2=run_van(A,T-1,modes[mi],1,got2,nf2);
            check(r2&&got2==expB&&nf2==expB.size(),"vanity CPU : fin de plage respectée");
        }
    }
    return ok;
}

static int vanity_main(int argc,char** argv){
    std::string prefix; bool have_pfx=false;
    long nbits=0; bool have_b=false, bb=false;
    bool have_r=false, r_colon=false; U256 rstart=ZERO, rend=ZERO;
    bool have_part=false; u64 part_i=0, part_n=0;
    int ncpu=usable_cpus(); int mode=MD_RANDOM; bool mode_set=false;   // vanity : ordre ALÉATOIRE par défaut (-s pour séquentiel)
    std::vector<RawMask> masks;
    bool have_pmf=false; u64 pmf_i=0, pmf_n=0;                 // -partmaskfile i:n
    std::string outdir, ckpt_user; bool no_ckpt=false, fresh=false, skip_test=false;
    double ckpt_every=120, time_limit=0, status_every=60;
    for(int i=1;i<argc;i++){
        std::string o=lower(argv[i]);
        auto need=[&](const char* nm)->const char*{ if(i+1>=argc){ fprintf(stderr,"%s doit être suivi d'une valeur.\n",nm); return nullptr; } return argv[++i]; };
        if(o=="-vanity"||o=="--vanity"){
            const char* v=need("--vanity"); if(!v) return 1;
            if(v[0]=='-'){ fprintf(stderr,"--vanity doit être suivi d'un préfixe Base58 (ex. --vanity 1Kids).\n"); return 1; }
            if(have_pfx){ fprintf(stderr,"--vanity ne peut être indiqué qu'une fois.\n"); return 1; }
            prefix=v; have_pfx=true; continue;
        }
        if(o=="-b"||o=="-bb"){
            const char* v=need(o.c_str()); if(!v) return 1;
            std::string sv=v; char* endp; long n=strtol(v,&endp,10);
            if(sv.empty()||*endp||n<1||n>255){ fprintf(stderr,"%s doit être suivi d'un nombre de bits entre 1 et 255 (ex. %s 70).\n",o.c_str(),o.c_str()); return 1; }
            if(have_b){ fprintf(stderr,"-b / -bb ne peuvent être indiqués qu'une fois.\n"); return 1; }
            nbits=n; have_b=true; bb=(o=="-bb"); continue;
        }
        if(o=="-r"){
            const char* v=need("-r"); if(!v) return 1;
            if(have_r){ fprintf(stderr,"-r ne peut être indiqué qu'une fois.\n"); return 1; }
            std::string ra=v; size_t sep=ra.find_first_of(":-");
            if(sep!=std::string::npos){
                std::string h1=van_clean_hex(ra.substr(0,sep)), h2=van_clean_hex(ra.substr(sep+1));
                if(!parse_hex(h1,rstart)||!parse_hex(h2,rend)){ fprintf(stderr,"Plage invalide : écris-la en hexadécimal, par exemple -r 400000:7fffff\n"); return 1; }
                r_colon=true;
            } else {
                if(!parse_hex(van_clean_hex(ra),rstart)){ fprintf(stderr,"Début de plage invalide : écris-le en hexadécimal, par exemple -r 400000 -b 22\n"); return 1; }
            }
            have_r=true; continue;
        }
        if(o=="-part"){
            const char* v=need("-part"); if(!v) return 1;
            std::string sv=v; size_t c=sv.find(':');
            std::string a=(c==std::string::npos)?"":sv.substr(0,c), b=(c==std::string::npos)?"":sv.substr(c+1);
            auto digits=[](const std::string& s){ return !s.empty()&&s.size()<=18&&s.find_first_not_of("0123456789")==std::string::npos; };
            if(!digits(a)||!digits(b)){ fprintf(stderr,"-part doit être de la forme i:n (ex. -part 3:400).\n"); return 1; }
            part_i=strtoull(a.c_str(),nullptr,10); part_n=strtoull(b.c_str(),nullptr,10);
            if(part_n<1||part_i<1||part_i>part_n){ fprintf(stderr,"-part i:n : il faut 1 <= i <= n (les parties sont numérotées de 1 à n).\n"); return 1; }
            have_part=true; continue;
        }
        int m=-1;
        if(o=="-s") m=MD_SEQ; else if(o=="-sobol") m=MD_SOBOL; else if(o=="-pi") m=MD_PI; else if(o=="-e") m=MD_E; else if(o=="-ln") m=MD_LN; else if(o=="-m") m=MD_MC;
        if(m>=0){
            if(mode_set){ fprintf(stderr,"Un seul mode à la fois (%s et %s indiqués).\n",MODE_NAME[mode],MODE_NAME[m]); return 1; }
            mode=m; mode_set=true; continue;
        }
        if(o=="-t"){
            const char* v=need("-t"); if(!v) return 1;
            std::string sv=v; if(sv.empty()||sv.find_first_not_of("0123456789")!=std::string::npos){ fprintf(stderr,"-t doit être suivi d'un nombre de cœurs.\n"); return 1; }
            ncpu=atoi(v); continue;
        }
        if(o=="-mask"){ const char* v=need("-mask"); if(!v) return 1; masks.push_back({v,false}); continue; }
        if(o=="-maskbin"){ const char* v=need("-maskbin"); if(!v) return 1; masks.push_back({v,true}); continue; }
        if(o=="-maskfile"){ const char* v=need("-maskfile"); if(!v) return 1;
            std::string e; if(!mask_read_file(v,masks,e)){ fprintf(stderr,"%s\n",e.c_str()); return 1; } continue; }
        if(o=="-partmaskfile"){ const char* v=need("-partmaskfile"); if(!v) return 1;
            if(have_pmf){ fprintf(stderr,"-partmaskfile ne peut être indiqué qu'une fois.\n"); return 1; }
            if(!parse_part_spec("-partmaskfile",v,pmf_i,pmf_n)) return 1; have_pmf=true; continue; }
        if(o=="--gpus"){ const char* v=need("--gpus"); if(!v) return 1; if(lower(v)!="none") fprintf(stderr,"Avertissement : --gpus ignoré (version CPU seule).\n"); continue; }
        if(o=="--no-gpu"){ continue; }                               // accepté pour compatibilité avec puzzle.cu (version CPU seule)
        if(o=="--out"){ const char* v=need("--out"); if(!v) return 1; outdir=v; continue; }
        if(o=="--ckpt"){ const char* v=need("--ckpt"); if(!v) return 1; ckpt_user=v; continue; }
        if(o=="--no-ckpt"){ no_ckpt=true; continue; }
        if(o=="--fresh"){ fresh=true; continue; }
        if(o=="--skip-test"){ skip_test=true; continue; }
        if(o=="--ckpt-every"){ const char* v=need("--ckpt-every"); if(!v) return 1; if(!parse_duration(v,ckpt_every)){ fprintf(stderr,"Durée invalide : %s\n",v); return 1; } continue; }
        if(o=="--status"){ const char* v=need("--status"); if(!v) return 1; if(!parse_duration(v,status_every)){ fprintf(stderr,"Durée invalide : %s\n",v); return 1; } continue; }
        if(o=="--time"){ const char* v=need("--time"); if(!v) return 1; if(!parse_duration(v,time_limit)){ fprintf(stderr,"Durée invalide : %s (ex. 3600, 90m, 11h30m)\n",v); return 1; } continue; }
        fprintf(stderr,"Argument inconnu en mode --vanity : %s\n",argv[i]); usage(argv[0]); return 1;
    }
    if(!have_pfx){ fprintf(stderr,"--vanity doit être suivi d'un préfixe (ex. --vanity 1Kids).\n"); return 1; }
    VanSpec vs; std::string err;
    if(!van_build(vs,prefix,err)){ fprintf(stderr,"Préfixe invalide : %s\n",err.c_str()); return 1; }

    // ---- plage de clés privées
    U256 start,end; bool range_given=true;
    if(have_r&&r_colon){
        if(have_b){ fprintf(stderr,"-r début:fin ne se combine pas avec -b / -bb.\n"); return 1; }
        start=rstart; end=rend;
        if(is_zero(start)){ fprintf(stderr,"Plage invalide : le début doit être au moins 1.\n"); return 1; }
        if(!geq(end,start)){ fprintf(stderr,"Plage invalide : le début doit être inférieur ou égal à la fin.\n"); return 1; }
        if(geq(end,CURVE_N)){ fprintf(stderr,"Plage invalide : la fin doit être inférieure à l'ordre de la courbe secp256k1.\n"); return 1; }
    } else if(have_r){                                              // -r début -b N  ->  [début, début + 2^N - 1]
        if(!have_b||bb){ fprintf(stderr,"-r début (sans :fin) doit être suivi de -b N (ex. -r 400000 -b 22).\n"); return 1; }
        if(is_zero(rstart)){ fprintf(stderr,"Plage invalide : le début doit être au moins 1.\n"); return 1; }
        start=rstart; U256 w=low_mask((int)nbits);
        if(add_raw(end,start,w)){ fprintf(stderr,"Plage invalide : début + 2^%ld - 1 dépasse 256 bits.\n",nbits); return 1; }
        if(geq(end,CURVE_N)){ fprintf(stderr,"Plage invalide : la fin doit être inférieure à l'ordre de la courbe secp256k1.\n"); return 1; }
    } else if(have_b){
        if(bb){ start=ONE; end=low_mask((int)nbits); }                // -bb N : [1, 2^N - 1]
        else { start=ZERO; start.d[(nbits-1)>>6]|=1ULL<<((nbits-1)&63); end=low_mask((int)nbits); }   // -b N : [2^(N-1), 2^N - 1]
    } else {
        start=ONE; sub_raw(end,CURVE_N,ONE); range_given=false;       // aucune plage : tout l'espace des clés privées
        if(!mode_set) mode=MD_RANDOM;
    }
    // ---- -part i:n : n parties de taille arrondie au-dessus, sans trou ni chevauchement ; seule la partie i est explorée
    U256 full_start=start, full_end=end;
    if(have_part){
        U256 total; sub_raw(total,end,start); add_small(total,1);
        U256 sz=div_small(total,part_n), chk=sz; u256_mul_small(chk,part_n);
        U256 rem; sub_raw(rem,total,chk); if(!is_zero(rem)) add_small(sz,1);
        U256 off=sz; u256_mul_small(off,part_i-1);
        U256 ps; add_raw(ps,start,off);
        if(!geq(end,ps)){ fprintf(stderr,"-part %llu:%llu : cette partie est vide (la plage ne contient pas assez de clés pour %llu parties).\n",
                                  (unsigned long long)part_i,(unsigned long long)part_n,(unsigned long long)part_n); return 1; }
        U256 szm1; sub_raw(szm1,sz,ONE); U256 pe; add_raw(pe,ps,szm1);
        if(!geq(end,pe)) pe=end;
        start=ps; end=pe;
    }
    if(ncpu>256) ncpu=256;

    if(ncpu<1) ncpu=1;
    if(!outdir.empty()&&outdir.back()!='/') outdir+="/";
    if(!outdir.empty()){ std::error_code ec; std::filesystem::create_directories(outdir,ec); }
    vs.out_path=outdir+"kangaroo_vanity.txt";

    build_tables();
    if(!skip_test){
        printf("Auto-tests en cours... "); fflush(stdout);
        if(!self_test()||!van_selftest()){ printf("ÉCHEC\n"); return 2; }
        printf("OK\n");
    }

    // ---- récapitulatif de la recherche
    U256 szr; sub_raw(szr,end,start); add_small(szr,1);
    const long double totalk=van_ld(szr);
    printf("Vanity : préfixe %s  (adresses P2PKH \"1...\", clé compressée, sensible à la casse)\n",vs.prefix.c_str());
    printf("Probabilité par clé : %.4Le  (1 clé sur %s)  -  %d intervalle(s) de hash160\n",vs.prob,van_big(1.0L/vs.prob).c_str(),vs.nint);
    {   auto trim=[](const U256& v){ std::string h=to_hex(v); size_t p=h.find_first_not_of('0'); return p==std::string::npos?std::string("0"):h.substr(p); };
        printf("Plage : %s -> %s  (%s clés)%s\n",trim(start).c_str(),trim(end).c_str(),van_big(totalk).c_str(),range_given?"":"  (aucune plage indiquée : tout l'espace des clés)"); }
    if(have_part){
        U256 ft; sub_raw(ft,full_end,full_start); add_small(ft,1);
        printf("Partie %llu / %llu de la plage %s -> %s  (%s clés au total)\n",(unsigned long long)part_i,(unsigned long long)part_n,
               short_hex(full_start).c_str(),short_hex(full_end).c_str(),fmt_si(van_ld(ft)).c_str());
    }
    printf("Adresses attendues sur cette plage : %.4Lg   (chance d'en trouver au moins une : %.2Lf %%)\n",totalk*vs.prob,100.0L*(-expm1l(-totalk*vs.prob)));
    if(vs.prob>1e-6L) printf("Attention : préfixe très court (1 clé sur %s) : les résultats seront très nombreux et ralentiront la recherche. 5 caractères ou plus sont conseillés.\n",fmt_si(1.0L/vs.prob).c_str());
    if(totalk<=ldexpl(1.0L,128)) printf("Attention : ces clés viennent d'une plage de seulement %s clés : une adresse générée ainsi peut être retrouvée par force brute. N'y laisse aucun fonds important.\n",fmt_si(totalk).c_str());

    signal(SIGINT,on_sigint); signal(SIGTERM,on_sigint);
    if(have_pmf){ std::string e; if(!mask_part_select(masks,pmf_i,pmf_n,e)){ fprintf(stderr,"%s\n",e.c_str()); return 1; } }
    static MaskSet g_mask_v; std::vector<std::string> mask_warns;
    if(!masks.empty()){
        std::string e; bool bin_fb=false;
        if(!mask_build_auto(g_mask_v,masks,start,end,0,mask_warns,e,bin_fb)){ fprintf(stderr,"%s\n",e.c_str()); return 1; }
        if(bin_fb) printf("Note : fichier de masques interprété en binaire (l'interprétation hexadécimale ne donnait aucun masque utilisable).\n");
        for(const std::string& w:mask_warns) fprintf(stderr,"Attention : %s\n",w.c_str());
    }
    static TargetSet vts; vts.addr=true; vts.nblocks=1; vts.bloom.assign(8,0); vts.list.clear();
    Run run;
    init_run_range(run,start,end,ncpu,&vts,true,mode,true,0,masks.empty()?nullptr:&g_mask_v);
    run.van=&vs;
    run.time_limit=time_limit; run.status_every=status_every<1?1:status_every; run.ckpt_every=ckpt_every<5?5:ckpt_every;
    if(!masks.empty()) mask_print_summary(g_mask_v,run,false);
    printf("Matériel : %d cœur(s) CPU\n",ncpu);
    printf("Résultats : %s  (adresse + clé privée hex + WIF)\n",vs.out_path.c_str());
    if(!no_ckpt){
        run.ckpt_path=ckpt_user.empty()? ckpt_default_path(run,outdir) : ckpt_user;
        if(!fresh){
            std::string msg; int rc=load_ckpt(run,run.ckpt_path,msg);
            if(rc==1){
                u64 nd=done_blocks(run);
                printf("Reprise : %s blocs déjà terminés sur %s, %s bloc(s) en cours, %s clés déjà testées  (%s)\n",
                    fmt_int(nd).c_str(),fmt_int(run.NB).c_str(),fmt_int(run.resume_q.size()).c_str(),fmt_keys((long double)run.keys_base).c_str(),run.ckpt_path.c_str());
            } else if(rc<0) printf("Point de reprise ignoré (%s) : %s\n",run.ckpt_path.c_str(),msg.c_str());
        }
        printf("Point de reprise : %s (toutes les %.0f s)\n",run.ckpt_path.c_str(),run.ckpt_every);
    }
    g_run=&run;
    run_engine(run,true);

    // ---- bilan
    const double el=std::chrono::duration<double>(clk::now()-run.t0).count();
    const bool interrupted=g_sigint.load();
    u64 nd=done_blocks(run);
    const bool complete=(nd==run.NB);
    printf("\n");
    bool all_saved=true;
    for(const Found& f:run.found) if(!f.saved) all_saved=false;   // les adresses ont déjà été annoncées une par une pendant la recherche : pas de ré-affichage
    printf("\n%zu adresse(s) trouvée(s) pour le préfixe %s en %s%s%s.\n",run.found.size(),vs.prefix.c_str(),fmt_time(el).c_str(),
           interrupted?" (interrompu)":"",run.timed_out?" (temps limite atteint)":"");
    if(complete) printf("Plage entièrement explorée (%s clés%s).\n",van_big(totalk).c_str(),have_part?(", partie "+std::to_string(part_i)+"/"+std::to_string(part_n)).c_str():"");
    else printf("Plage explorée à %.2Lf %%.\n",100.0L*std::min<long double>(1.0L,((long double)run.keys_base+[&]{ u64 k=0; for(auto& s:run.st) k+=s.keys.load(); return (long double)k; }())/totalk));
    if((interrupted||run.timed_out)&&!run.ckpt_path.empty()) printf("Point de reprise écrit : relance la même commande pour continuer.\n");
    if(!run.found.empty()&&all_saved) printf("Résultats enregistrés (et relus pour vérification) : %s\n",vs.out_path.c_str());
    if(!all_saved){ fprintf(stderr,"\n*** ATTENTION : au moins une adresse n'a pas pu être enregistrée dans un fichier ! ***\n"); return 4; }
    return interrupted?130:0;
}

static void usage(const char* prog){
    fprintf(stderr,
        "Usage : %s <cible> <bits | début:fin> [address] [mode] [options]\n"
        "   ou : %s -pN [mode] [options]      (puzzle Bitcoin n°N, 1 à 80)\n"
        "  -pN           : puzzle Bitcoin n°N : adresse intégrée et plage 2^(N-1)…2^N-1 automatique (ex. -p66 -sobol ; remplace <cible> <bits>)\n"
        "  <bits | début:fin> : nombre de bits, ou plage de clés en hexadécimal (ex. 2000000:3ffffff)\n"
        "  <cible> : clé publique compressée, adresse, fichier .txt (une cible par ligne) ou fichier .bloom\n"
        "  address : à ajouter quand les cibles sont des adresses Bitcoin (1... ou bc1q...)\n"
        "  mode (un seul) : -r aléatoire (défaut) | -s séquentiel | -sobol | -pi | -e | -ln | -m Monte Carlo\n"
        "  -vN          : permutation : chaque clé générée est suivie de toutes les valeurs de ses N derniers bits (ex. -sobol -v10 ; N de 1 à 62)\n"
        "  -mask M       : masque de clé privée en hexadécimal, ? pour un chiffre inconnu (ex. -mask 2a??????????????????) ; répétable\n"
        "  -maskbin M    : comme -mask, mais M est en binaire (un ? par bit)\n"
        "  -maskfile F   : fichier texte, un masque hex ou bin: par ligne (# commentaire) ; -mask/-maskbin/-maskfile se combinent, pas avec -v\n"
        "  -partmaskfile i:n : ne garde que la partie i sur n de la liste de masques (découpée dans l'ordre, parties égales à un masque près ; ex. 2:70)\n"
        "  -t N          : nombre de cœurs utilisés (par défaut : tous les cœurs utilisables)\n"
        "  --out DIR     : dossier des résultats et du point de reprise\n"
        "  --ckpt FILE | --no-ckpt | --fresh | --ckpt-every S : point de reprise (relancer la même commande reprend la recherche)\n"
        "  --time T      : arrêt propre après T (ex. 3600, 90m, 11h30m) ; le point de reprise est écrit\n"
        "  --status S    : intervalle de la ligne de récapitulatif (défaut 60 s)     --skip-test : saute les auto-tests\n"
        "  --info | --selftest | --genpub <clé_hex> | --genaddr <clé_hex>\n"
        "\n"
        "Mode vanity (adresses P2PKH \"1...\" personnalisées, résultats dans kangaroo_vanity.txt) :\n"
        "   %s --vanity <préfixe> [plage] [-part i:n] [options]\n"
        "  --vanity P    : préfixe Base58 sensible à la casse (\"1\" initial facultatif), ex. --vanity 1Kids\n"
        "  plage         : -b N -> [2^(N-1), 2^N-1] | -bb N -> [1, 2^N-1] | -r début:fin (hexadécimal) | -r début -b N -> début + 2^N - 1\n"
        "  -part i:n     : découpe la plage en n parties et n'explore que la partie i (1 <= i <= n), ex. -b 70 -part 3:400\n"
        "  (ordre : aléatoire par défaut, ou -s séquentiel -sobol -pi -e -ln -m ; -t, --out, --time, --ckpt... comme ci-dessus)\n",prog,prog,prog);
}
static void print_info(){
    printf("CPU : %d cœur(s) utilisables\n",usable_cpus());
    printf("GPU : cette version est CPU seul (le GPU est géré par puzzle.cu)\n");
}

int main(int argc,char** argv){
#ifdef _WIN32
    SetConsoleOutputCP(65001);   // accents corrects dans cmd.exe
    { HANDLE h=GetStdHandle(STD_OUTPUT_HANDLE); DWORD m=0;
      if(GetConsoleMode(h,&m)&&SetConsoleMode(h,m|0x0004)) g_ansi=true; }   // couleurs ANSI (Windows 10+)
#else
    g_ansi=isatty(1)||getenv("PUZZLE_ANSI");
#endif
    std::set_terminate(on_terminate);
#ifdef _WIN32
    SetUnhandledExceptionFilter(crash_filter);
#endif
    init_curve();
    if(argc==3 && (std::string(argv[1])=="--genpub"||std::string(argv[1])=="--genaddr")){
        U256 k; if(!parse_hex(argv[2],k)||is_zero(k)){ fprintf(stderr,"Clé invalide\n"); return 1; }
        Aff P=aff_mul(k,GEN);
        if(std::string(argv[1])=="--genpub") printf("%s\n",compress_hex(P).c_str());
        else { Ent e=ent_of_point(true,P); printf("%s\n%s\n",ent_addr_legacy(e).c_str(),ent_addr_bech32(e).c_str()); }
        return 0;
    }
    if(argc==2&&std::string(argv[1])=="--info"){ print_info(); return 0; }
    for(int i=1;i<argc;i++){ std::string a=lower(argv[i]); if(a=="-vanity"||a=="--vanity") return vanity_main(argc,argv); }
    if(argc==2&&std::string(argv[1])=="--selftest"){
        printf("Auto-tests en cours... "); fflush(stdout);
        build_tables();
        bool ok=self_test(); printf(ok?"OK\n":"ÉCHEC\n"); return ok?0:2;
    }
    std::vector<std::string> pz_store; std::vector<char*> pz_ptrs;
    if(expand_puzzle_arg(argc,argv,pz_store,pz_ptrs)<0) return 1;      // -pN : puzzle Bitcoin n°N
    if(argc<3){ usage(argv[0]); return 1; }
    std::string arg=argv[1];
    long bits=0; bool range_given=false; U256 rstart=ZERO, rend=ZERO;
    {
        std::string ra=argv[2]; size_t sep=ra.find_first_of(":-");
        if(sep!=std::string::npos){
            auto clean=[](std::string h){ if(h.size()>=2&&h[0]=='0'&&(h[1]=='x'||h[1]=='X')) h=h.substr(2); return h; };
            std::string h1=clean(ra.substr(0,sep)), h2=clean(ra.substr(sep+1));
            if(!parse_hex(h1,rstart)||!parse_hex(h2,rend)){ fprintf(stderr,"Plage invalide : écris-la en hexadécimal, par exemple 2000000:3ffffff\n"); return 1; }
            if(is_zero(rstart)){ fprintf(stderr,"Plage invalide : le début doit être au moins 1.\n"); return 1; }
            if(!geq(rend,rstart)){ fprintf(stderr,"Plage invalide : le début doit être inférieur ou égal à la fin.\n"); return 1; }
            if(geq(rend,CURVE_N)){ fprintf(stderr,"Plage invalide : la fin doit être inférieure à l'ordre de la courbe secp256k1.\n"); return 1; }
            range_given=true;
        } else {
            char* endp; bits=strtol(argv[2],&endp,10);
            if(*endp||bits<1||bits>255){ fprintf(stderr,"Nombre de bits invalide (1 à 255), ou plage début:fin en hexadécimal.\n"); return 1; }
        }
    }
    bool addrmode=false; int nthreads=usable_cpus(); int mode=MD_RANDOM; bool mode_set=false; u64 Rv=0;
    std::vector<RawMask> masks;
    bool have_pmf=false; u64 pmf_i=0, pmf_n=0;                 // -partmaskfile i:n
    std::string outdir, ckpt_user; bool no_ckpt=false, fresh=false, skip_test=false; double ckpt_every=120, time_limit=0, status_every=60;
    for(int i=3;i<argc;i++){
        std::string o=lower(argv[i]);
        int m=-1;
        auto need=[&](const char* nm)->const char*{ if(i+1>=argc){ fprintf(stderr,"%s doit être suivi d'une valeur.\n",nm); return nullptr; } return argv[++i]; };
        if(o=="address"){ addrmode=true; continue; }
        else if(o=="-r") m=MD_RANDOM; else if(o=="-s") m=MD_SEQ; else if(o=="-sobol") m=MD_SOBOL;
        else if(o=="-pi") m=MD_PI; else if(o=="-e") m=MD_E; else if(o=="-ln") m=MD_LN; else if(o=="-m") m=MD_MC;
        if(m>=0){
            if(mode_set){ fprintf(stderr,"Un seul mode à la fois (%s et %s indiqués).\n",MODE_NAME[mode],MODE_NAME[m]); usage(argv[0]); return 1; }
            mode=m; mode_set=true; continue;
        }
        if(o.size()>=2&&o[0]=='-'&&o[1]=='v'&&(o.size()==2||o.find_first_not_of("0123456789",2)==std::string::npos)){     // -vN ou -v N
            std::string a2=o.size()>2? o.substr(2) : ((i+1<argc)?argv[i+1]:"");
            if(a2.empty()||a2.size()>2||a2.find_first_not_of("0123456789")!=std::string::npos||strtoull(a2.c_str(),nullptr,10)==0||strtoull(a2.c_str(),nullptr,10)>(u64)V_MAXBITS){
                fprintf(stderr,"-v doit être suivi du nombre N de bits faibles à permuter, entre 1 et %d (ex. -v10).\n",V_MAXBITS); return 1; }
            Rv=strtoull(a2.c_str(),nullptr,10); if(o.size()==2) i++; continue;
        }
        if(o=="-mask"){ const char* v=need("-mask"); if(!v) return 1; masks.push_back({v,false}); continue; }
        if(o=="-maskbin"){ const char* v=need("-maskbin"); if(!v) return 1; masks.push_back({v,true}); continue; }
        if(o=="-maskfile"){ const char* v=need("-maskfile"); if(!v) return 1;
            std::string e; if(!mask_read_file(v,masks,e)){ fprintf(stderr,"%s\n",e.c_str()); return 1; } continue; }
        if(o=="-partmaskfile"){ const char* v=need("-partmaskfile"); if(!v) return 1;
            if(have_pmf){ fprintf(stderr,"-partmaskfile ne peut être indiqué qu'une fois.\n"); return 1; }
            if(!parse_part_spec("-partmaskfile",v,pmf_i,pmf_n)) return 1; have_pmf=true; continue; }
        if(o=="-t"){
            if(i+1>=argc||std::string(argv[i+1]).find_first_not_of("0123456789")!=std::string::npos||std::string(argv[i+1]).empty()){
                fprintf(stderr,"-t doit être suivi d'un nombre de cœurs.\n"); return 1; }
            nthreads=atoi(argv[++i]); continue;
        }
        if(o=="--gpus"){ const char* v=need("--gpus"); if(!v) return 1; if(lower(v)!="none") fprintf(stderr,"Avertissement : --gpus ignoré (version CPU seule).\n"); continue; }
        if(o=="--no-gpu"){ continue; }                               // accepté pour compatibilité avec puzzle.cu (version CPU seule)
        if(o=="--out"){ const char* v=need("--out"); if(!v) return 1; outdir=v; continue; }
        if(o=="--ckpt"){ const char* v=need("--ckpt"); if(!v) return 1; ckpt_user=v; continue; }
        if(o=="--no-ckpt"){ no_ckpt=true; continue; }
        if(o=="--fresh"){ fresh=true; continue; }
        if(o=="--skip-test"){ skip_test=true; continue; }
        if(o=="--ckpt-every"){ const char* v=need("--ckpt-every"); if(!v) return 1; if(!parse_duration(v,ckpt_every)){ fprintf(stderr,"Durée invalide : %s\n",v); return 1; } continue; }
        if(o=="--status"){ const char* v=need("--status"); if(!v) return 1; if(!parse_duration(v,status_every)){ fprintf(stderr,"Durée invalide : %s\n",v); return 1; } continue; }
        if(o=="--time"){ const char* v=need("--time"); if(!v) return 1; if(!parse_duration(v,time_limit)){ fprintf(stderr,"Durée invalide : %s (ex. 3600, 90m, 11h30m)\n",v); return 1; } continue; }
        if(!o.empty()&&o.find_first_not_of("0123456789")==std::string::npos){ nthreads=atoi(o.c_str()); continue; }   // ancien format : nombre seul
        fprintf(stderr,"Argument inconnu : %s\n",argv[i]); usage(argv[0]); return 1;
    }
    if(!outdir.empty()&&outdir.back()!='/'&&outdir.back()!='\\') outdir+="/";
    if(!outdir.empty()){ std::error_code ec; std::filesystem::create_directories(outdir,ec); }

    // ---- chargement des cibles
    TargetSet ts; ts.addr=addrmode;
    Ent single;
    bool is_single_pub=!addrmode&&looks_like_pub(arg)&&parse_pub(arg,single);
    int addr_rc=addrmode?parse_address(arg,single):0;
    if(!addrmode&&looks_like_pub(arg)&&!is_single_pub){ fprintf(stderr,"Clé publique invalide ou hors de la courbe secp256k1.\n"); return 1; }
    if(is_single_pub||(addrmode&&addr_rc==1)){
        ts.list.push_back(single); finalize_targets(ts);
        if(addrmode) printf("Cible unique : %s  (hash160 %s)\n",ent_addr_legacy(single).c_str(),ent_hash_hex(single).c_str());
        else printf("Cible unique : %s\n",ent_hex(single).c_str());
    } else if(addrmode&&addr_rc==2){
        fprintf(stderr,"Adresse valide mais non supportée (seules les adresses P2PKH '1...' et P2WPKH 'bc1q...' le sont).\n"); return 1;
    } else {
        bool isbloom=arg.size()>=6&&arg.substr(arg.size()-6)==".bloom";
        if(isbloom){
            std::string err; TargetSet tmp;
            if(!load_bloom(arg,tmp,err)){ fprintf(stderr,"Impossible de charger %s : %s\n",arg.c_str(),err.c_str()); return 1; }
            if(tmp.addr!=addrmode){
                fprintf(stderr,tmp.addr?"Ce filtre contient des adresses : ajoute l'argument address.\n"
                                       :"Ce filtre contient des clés publiques : retire l'argument address.\n");
                return 1;
            }
            ts=tmp;
            printf("Filtre chargé : %s  (%s %s)\n",arg.c_str(),fmt_int(ts.list.size()).c_str(),ts.addr?"adresses":"clés publiques");
        } else {
            std::string txt;
            if(!read_file(arg,txt)){
                if(!addrmode&&(parse_address(arg,single)!=0))
                    fprintf(stderr,"Cette cible est une adresse Bitcoin : ajoute l'argument address.\n");
                else fprintf(stderr,"Impossible de lire %s (ni clé publique, ni adresse, ni fichier valide).\n",arg.c_str());
                return 1;
            }
            u64 h=fnv64(txt);
            size_t dot=arg.find_last_of('.'), slash=arg.find_last_of("/\\");
            std::string bname=((dot==std::string::npos||(slash!=std::string::npos&&dot<slash))?arg:arg.substr(0,dot))+".bloom";
            std::string err; TargetSet old;
            if(load_bloom(bname,old,err)&&old.src_hash==h&&old.addr==addrmode){
                ts=old; printf("Filtre existant réutilisé : %s  (%s cibles)\n",bname.c_str(),fmt_int(ts.list.size()).c_str());
            } else {
                u64 lines,rej,dups,unsup; parse_targets_text(txt,ts,addrmode,lines,rej,dups,unsup); ts.src_hash=h;
                if(ts.list.empty()){
                    fprintf(stderr,"Aucune cible valide dans %s (%s lignes lues).%s\n",arg.c_str(),fmt_int(lines).c_str(),
                        addrmode?"":" Si ce sont des adresses, ajoute l'argument address.");
                    return 1;
                }
                printf("Fichier lu : %s lignes, %s valides, %s doublons, %s rejetées",
                    fmt_int(lines).c_str(),fmt_int(ts.list.size()).c_str(),fmt_int(dups).c_str(),fmt_int(rej).c_str());
                if(unsup) printf(", %s non supportées (P2SH/P2WSH/Taproot)",fmt_int(unsup).c_str());
                printf("\n");
                if(save_bloom(ts,bname)) printf("Filtre enregistré : %s\n",bname.c_str());
                else fprintf(stderr,"Attention : impossible d'écrire %s\n",bname.c_str());
            }
        }
    }

    build_tables();
    if(!skip_test){
        printf("Auto-tests en cours... "); fflush(stdout);
        if(!self_test()){ printf("ÉCHEC\n"); return 2; }
        printf("OK\n");
    }

    if(nthreads<1) nthreads=1;
    if(nthreads>256) nthreads=256;

    signal(SIGINT,on_sigint); signal(SIGTERM,on_sigint);
    Run run;
    if(Rv&&!masks.empty()){ fprintf(stderr,"-v et -mask/-maskbin/-maskfile ne se combinent pas.\n"); return 1; }
    if(have_pmf){ std::string e; if(!mask_part_select(masks,pmf_i,pmf_n,e)){ fprintf(stderr,"%s\n",e.c_str()); return 1; } }
    static MaskSet g_mask; std::vector<std::string> mask_warns;
    if(!masks.empty()){
        if(!range_given){ rstart=ZERO; rstart.d[(bits-1)>>6]|=1ULL<<((bits-1)&63); rend=low_mask((int)bits); range_given=true; }
        std::string e; bool bin_fb=false;
        if(!mask_build_auto(g_mask,masks,rstart,rend,0,mask_warns,e,bin_fb)){ fprintf(stderr,"%s\n",e.c_str()); return 1; }
        if(bin_fb) printf("Note : fichier de masques interprété en binaire (l'interprétation hexadécimale ne donnait aucun masque utilisable).\n");
        for(const std::string& w:mask_warns) fprintf(stderr,"Attention : %s\n",w.c_str());
    }
    if(range_given) init_run_range(run,rstart,rend,nthreads,&ts,true,mode,true,Rv,masks.empty()?nullptr:&g_mask); else init_run(run,(int)bits,nthreads,&ts,true,mode,Rv);
    if(run.r_fail){ fprintf(stderr,"Permutation -v%llu impossible : plus de 2^63 blocs pour cette plage (augmente N).\n",(unsigned long long)Rv); return 1; }
    run.out_main=outdir+"cle_trouvee.txt"; run.out_copy=outdir+"cle_trouvee_copie.txt";
    run.time_limit=time_limit; run.status_every=status_every<1?1:status_every; run.ckpt_every=ckpt_every<5?5:ckpt_every;
    if(!masks.empty()) mask_print_summary(g_mask,run,false);
    if(Rv){
        printf("Permutation : chaque clé générée est suivie de ses %s valeurs des %d derniers bits (blocs alignés de 2^%d clés) : %s blocs.\n",fmt_int(1ULL<<run.vbits).c_str(),run.vbits,run.vbits,fmt_int(run.NB).c_str());
        if(run.sparse) printf("Plus de 2^26 blocs : suite calculée sans mémoire (-s et -sobol restent exhaustifs ; les autres modes tirent sans exclure les blocs déjà faits).\n");
        if(run.vbits<10) printf("Attention : blocs de moins de 1024 clés : le calcul par lots est sous-utilisé (plus lent). N >= 10 conseillé.\n");
    }
    printf("Matériel : %d cœur(s) CPU\n",nthreads);
    if(!no_ckpt){
        run.ckpt_path=ckpt_user.empty()? ckpt_default_path(run,outdir) : ckpt_user;
        if(!fresh){
            std::string msg; int rc=load_ckpt(run,run.ckpt_path,msg);
            if(rc==1){
                u64 nd=done_blocks(run);
                printf("Reprise : %s blocs déjà terminés sur %s, %s bloc(s) en cours, %s clés déjà testées  (%s)\n",
                    fmt_int(nd).c_str(),fmt_int(run.NB).c_str(),fmt_int(run.resume_q.size()).c_str(),fmt_keys((long double)run.keys_base).c_str(),run.ckpt_path.c_str());
            } else if(rc<0) printf("Point de reprise ignoré (%s) : %s\n",run.ckpt_path.c_str(),msg.c_str());
        }
        printf("Point de reprise : %s (toutes les %.0f s)\n",run.ckpt_path.c_str(),run.ckpt_every);
    }
    g_run=&run;
    run_engine(run,true);

    // ---- récapitulatif
    double el=std::chrono::duration<double>(clk::now()-run.t0).count();
    bool interrupted=g_sigint.load();
    printf("\n");
    bool all_saved=true;
    for(const Found& f:run.found){
        if(ts.addr)
            printf("  ✔ %s  /  %s\n      clé publique : %s\n",ent_addr_legacy(f.ent).c_str(),ent_addr_bech32(f.ent).c_str(),f.pub.c_str());
        else printf("  ✔ %s\n",f.pub.c_str());
        printf("      clé privée   : %s\n      décimal      : %s\n",to_hex(f.key).c_str(),to_dec(f.key).c_str());
        if(f.outside) printf("      NB : clé valide, mais juste après la fin de la plage demandée\n");
        if(f.saved) printf("      enregistrée  : %s\n",f.path.c_str());
        else { all_saved=false; printf("      !!! NON ENREGISTRÉE : recopie cette clé tout de suite !!!\n"); }
    }
    printf("\n%zu / %zu cible(s) trouvée(s) en %s%s%s.\n",run.found.size(),ts.list.size(),fmt_time(el).c_str(),
           interrupted?" (interrompu)":"",run.timed_out?" (temps limite atteint)":"");
    if((interrupted||run.timed_out)&&!run.ckpt_path.empty()) printf("Point de reprise écrit : relance la même commande pour continuer.\n");
    if(!run.found.empty()&&all_saved) printf("Résultats enregistrés (et relus pour vérification) : %s\n",run.found.back().path.c_str());
    if(!all_saved){ fprintf(stderr,"\n*** ATTENTION : au moins une clé n'a pas pu être enregistrée dans un fichier ! ***\n"); return 4; }
    return run.found.size()==ts.list.size()?0:(interrupted?130:1);
}
