import { useEffect, useState, type ReactNode } from 'react';

export const STAGE_W = 1600;
export const STAGE_H = 900;

/** Fixed 16:9 stage, scaled to fit the viewport and centered (letterboxed). */
export function Stage({ children }: { children: ReactNode }) {
  const [scale, setScale] = useState(1);

  useEffect(() => {
    const update = () =>
      setScale(Math.min(window.innerWidth / STAGE_W, window.innerHeight / STAGE_H));
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  return (
    <div className="fixed inset-0 grid place-items-center">
      <div
        className="relative overflow-hidden bg-paper shadow-2xl"
        style={{ width: STAGE_W, height: STAGE_H, transform: `scale(${scale})` }}
      >
        {children}
      </div>
    </div>
  );
}
