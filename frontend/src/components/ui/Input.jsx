import { forwardRef } from 'react';
import { cn } from '../../utils/cn';

export const Input = forwardRef(function Input(
  { className, icon: Icon, ...props },
  ref,
) {
  return (
    <div className="relative w-full">
      {Icon && (
        <Icon className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted pointer-events-none" />
      )}
      <input
        ref={ref}
        className={cn(
          'w-full h-10 rounded-lg bg-bg-subtle border border-border text-sm text-text-primary placeholder:text-text-muted',
          'px-3.5 focus-ring focus:border-accent-cyan/50 transition-colors',
          Icon && 'pl-9',
          className,
        )}
        {...props}
      />
    </div>
  );
});

export const Select = forwardRef(function Select(
  { className, children, ...props },
  ref,
) {
  return (
    <select
      ref={ref}
      className={cn(
        'w-full h-10 rounded-lg bg-bg-subtle border border-border text-sm text-text-primary',
        'px-3.5 focus-ring focus:border-accent-cyan/50 transition-colors appearance-none cursor-pointer',
        className,
      )}
      {...props}
    >
      {children}
    </select>
  );
});

export const Checkbox = forwardRef(function Checkbox(
  { className, label, ...props },
  ref,
) {
  return (
    <label className="flex items-center gap-2.5 cursor-pointer group">
      <span className="relative flex items-center justify-center">
        <input
          ref={ref}
          type="checkbox"
          className={cn(
            'peer w-4 h-4 rounded border border-border bg-bg-subtle appearance-none',
            'checked:bg-accent-cyan checked:border-accent-cyan transition-colors cursor-pointer focus-ring',
            className,
          )}
          {...props}
        />
        <svg
          className="pointer-events-none absolute w-3 h-3 text-bg opacity-0 peer-checked:opacity-100"
          fill="none" stroke="currentColor" strokeWidth="3" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
        </svg>
      </span>
      <span className="text-sm text-text-secondary group-hover:text-text-primary transition-colors">
        {label}
      </span>
    </label>
  );
});
