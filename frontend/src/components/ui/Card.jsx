import { cn } from '../../utils/cn';

export function Card({ className, children, ...props }) {
  return (
    <div className={cn('glass glass-hover', className)} {...props}>
      {children}
    </div>
  );
}

export function CardHeader({ className, children, ...props }) {
  return (
    <div
      className={cn('flex items-center justify-between gap-3 px-5 py-4 border-b border-border', className)}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardTitle({ className, children, icon: Icon, ...props }) {
  return (
    <h3
      className={cn('flex items-center gap-2 text-sm font-semibold text-text-primary', className)}
      {...props}
    >
      {Icon && <Icon className="w-4 h-4 text-accent-cyan" />}
      {children}
    </h3>
  );
}

export function CardBody({ className, children, ...props }) {
  return (
    <div className={cn('p-5', className)} {...props}>
      {children}
    </div>
  );
}
