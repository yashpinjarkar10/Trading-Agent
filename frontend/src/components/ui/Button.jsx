import { forwardRef } from 'react';
import { cn } from '../../utils/cn';

const variants = {
  primary:
    'bg-gradient-to-r from-accent-cyan to-accent-violet text-bg font-semibold hover:opacity-90 shadow-glow',
  secondary:
    'bg-bg-subtle border border-border hover:border-border-strong text-text-primary',
  ghost:
    'text-text-secondary hover:text-text-primary hover:bg-white/5',
  danger:
    'bg-accent-red/10 text-accent-red border border-accent-red/30 hover:bg-accent-red/20',
};

const sizes = {
  sm: 'h-8 px-3 text-xs gap-1.5',
  md: 'h-10 px-4 text-sm gap-2',
  lg: 'h-12 px-6 text-base gap-2',
  icon: 'h-9 w-9 p-0 justify-center',
};

const Button = forwardRef(function Button(
  { variant = 'primary', size = 'md', className, children, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      className={cn(
        'inline-flex items-center justify-center rounded-lg transition-all duration-150 focus-ring disabled:opacity-50 disabled:pointer-events-none whitespace-nowrap',
        variants[variant],
        sizes[size],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
});

export default Button;
