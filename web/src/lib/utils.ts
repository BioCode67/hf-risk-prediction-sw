import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/** shadcn/ui's class helper — kept at this path so generated components import it unchanged. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
