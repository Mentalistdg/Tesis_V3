import logoImg from '../assets/logo.png';

interface LoadingScreenProps {
  message?: string;
}

export default function LoadingScreen({ message = 'Loading data...' }: LoadingScreenProps) {
  return (
    <div className="fixed inset-0 z-50 bg-black flex flex-col items-center justify-center">

      {/* Main content — logo + spinner */}
      <div className="flex flex-col items-center gap-6">
        <img
          src={logoImg}
          alt="CRONOS"
          className="h-20 w-auto opacity-90"
        />
        <div className="loading-spinner" />
        <p className="text-sm text-[#525252] tracking-wide">{message}</p>
      </div>

      {/* Bottom attribution */}
      <div className="absolute bottom-8 flex flex-col items-center gap-3">
        <div className="w-16 h-px bg-gradient-to-r from-transparent via-[#c41e3a]/40 to-transparent" />
        <div className="flex flex-col items-center gap-1.5">
          <p className="text-xs text-[#737373] tracking-[0.15em] uppercase">
            Tesis de Grado &mdash; Mag&iacute;ster en Finanzas
          </p>
          <p className="text-[11px] text-[#5a5a5a] tracking-[0.08em] leading-relaxed text-center max-w-md italic">
            Modelado Cuantitativo para la Predicci&oacute;n Direccional y Generaci&oacute;n de Alfa en el S&amp;P 500
          </p>
        </div>
        <p className="text-[11px] text-[#636363] tracking-[0.2em] uppercase">
          David Gonz&aacute;lez Ca&ntilde;&oacute;n &middot; 2026
        </p>
      </div>
    </div>
  );
}
