import { cn } from '../../utils/cn';

const tones = {
  neutral: 'border-border bg-bg-subtle text-text-secondary',
  cyan: 'border-accent-cyan/30 bg-accent-cyan/10 text-accent-cyan',
  violet: 'border-accent-violet/30 bg-accent-violet/10 text-accent-violet',
  green: 'border-accent-green/30 bg-accent-green/10 text-accent-green',
  red: 'border-accent-red/30 bg-accent-red/10 text-accent-red',
  amber: 'border-accent-amber/30 bg-accent-amber/10 text-accent-amber',
};

export default function Badge({ tone = 'neutral', className, children, dot = false, ...props }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium border',
        tones[tone],
        className,
      )}
      {...props}
    >
      {dot && (
        <span className={cn('w-1.5 h-1.5 rounded-full animate-pulse-soft', `bg-current`)} />
      )}
      {children}
    </span>
  );
}
